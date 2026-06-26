import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(ROOT_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from util import get_explainable_method, get_torchvision_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate batch attack outputs with mean Attack Success Rate, "
            "Margin Loss, Saliency Loss, and Spearman rank correlation"
        )
    )
    parser.add_argument(
        "--report-path",
        type=str,
        required=True,
        help="Path to batch_report.json produced by run_batch.py",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for saliency recomputation",
    )
    parser.add_argument(
        "--explain-method",
        type=str,
        default=None,
        choices=["simple_gradient", "integrated_gradients"],
        help="Override explain method. If omitted, infer from report approach tag",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only first N successful samples",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Where to save evaluation JSON. Default: <run_root>/evaluation_summary.json",
    )
    return parser.parse_args()


def _extract_explain_method(approach_tag):
    if not isinstance(approach_tag, str):
        return None
    for part in approach_tag.split("__"):
        if part.startswith("exp-"):
            return part[len("exp-") :]
    return None


def _rankdata_average_ties(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]

    n = values.shape[0]
    ranks_sorted = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks_sorted[i:j] = avg_rank
        i = j

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _spearman_rank_corr(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("Spearman inputs must have the same number of elements")
    if x.size < 2:
        return float("nan")

    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = math.sqrt(float(np.dot(rx, rx)) * float(np.dot(ry, ry)))
    if denom < 1e-12:
        return float("nan")

    return float(np.dot(rx, ry) / denom)


def _load_image_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    return T.ToTensor()(image).unsqueeze(0)


def _safe_mean(values):
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def _compute_spearman_for_sample(model, normalize, explain_fn, result, device):
    output_dir = Path(result["output_dir"])
    clean_path = output_dir / "clean.png"
    adv_path = output_dir / "adv.png"

    if not clean_path.exists() or not adv_path.exists():
        return None, "missing_clean_or_adv_image"

    label = int(result.get("true_label", result.get("clean_pred", -1)))
    if label < 0:
        return None, "missing_label"

    y_true = torch.tensor([label], device=device)

    clean_x = _load_image_tensor(clean_path).to(device)
    adv_x = _load_image_tensor(adv_path).to(device)

    clean_map, _ = explain_fn(model, clean_x, normalize, y_true)
    adv_map, _ = explain_fn(model, adv_x, normalize, y_true)

    corr = _spearman_rank_corr(
        clean_map.detach().float().cpu().numpy(),
        adv_map.detach().float().cpu().numpy(),
    )
    return corr, None


def main():
    args = parse_args()

    report_path = Path(args.report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"report file not found: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    model_name = report.get("model")
    if not model_name:
        raise ValueError("batch_report.json is missing 'model'")

    explain_method = args.explain_method or _extract_explain_method(report.get("approach"))
    if explain_method is None:
        explain_method = "simple_gradient"

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is not available, switching to CPU")
        args.device = "cpu"

    device = torch.device(args.device)

    model, _, normalize = get_torchvision_model(model_name, pretrained=True)
    model = model.to(device)
    model.eval()

    explain_fn = get_explainable_method(explain_method)

    ok_results = [item for item in report.get("results", []) if item.get("status") == "ok"]
    if args.max_samples is not None:
        ok_results = ok_results[: args.max_samples]

    if not ok_results:
        raise ValueError("No successful samples (status='ok') found in report")

    margin_losses = []
    saliency_losses = []
    success_flags = []
    spearman_scores = []
    spearman_failed = 0

    for result in tqdm(ok_results, desc="Evaluating", unit="img"):
        true_label = int(result.get("true_label", result.get("clean_pred", -1)))
        adv_pred = int(result.get("adv_pred", -1))

        margin_losses.append(float(result.get("margin_loss", float("nan"))))
        saliency_losses.append(float(result.get("saliency_loss", float("nan"))))

        success_flags.append(int(adv_pred != true_label and true_label >= 0 and adv_pred >= 0))

        corr, err = _compute_spearman_for_sample(
            model=model,
            normalize=normalize,
            explain_fn=explain_fn,
            result=result,
            device=device,
        )
        if err is not None:
            spearman_failed += 1
        spearman_scores.append(corr)

    attack_success_rate = float(np.mean(success_flags))
    mean_margin_loss = _safe_mean(margin_losses)
    mean_saliency_loss = _safe_mean(saliency_losses)
    mean_spearman = _safe_mean(spearman_scores)

    summary = {
        "report_path": str(report_path),
        "model": model_name,
        "approach": report.get("approach"),
        "explain_method": explain_method,
        "num_ok_samples_evaluated": len(ok_results),
        "attack_success_rate": attack_success_rate,
        "mean_margin_loss": mean_margin_loss,
        "mean_saliency_loss": mean_saliency_loss,
        "mean_spearman_adv_vs_clean_saliency": mean_spearman,
        "spearman_failed_samples": spearman_failed,
    }

    output_file = Path(args.output_file) if args.output_file else report_path.parent / "evaluation_summary.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== Evaluation summary ===")
    print(f"report_path: {summary['report_path']}")
    print(f"model: {summary['model']}")
    print(f"approach: {summary['approach']}")
    print(f"explain_method: {summary['explain_method']}")
    print(f"num_ok_samples_evaluated: {summary['num_ok_samples_evaluated']}")
    print(f"attack_success_rate: {summary['attack_success_rate']:.6f}")
    print(f"mean_margin_loss: {summary['mean_margin_loss']:.6f}")
    print(f"mean_saliency_loss: {summary['mean_saliency_loss']:.6f}")
    print(
        "mean_spearman_adv_vs_clean_saliency: "
        f"{summary['mean_spearman_adv_vs_clean_saliency']:.6f}"
    )
    print(f"spearman_failed_samples: {summary['spearman_failed_samples']}")
    print(f"saved_to: {output_file}")


if __name__ == "__main__":
    main()
