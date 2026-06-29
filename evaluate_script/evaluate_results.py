import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


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
        default=None,
        help="Path to batch_report.json produced by run_batch.py",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help=(
            "Path to directory containing multiple run folders. "
            "Each run folder should have batch_report.json"
        ),
    )
    parser.add_argument(
        "--print-latex-lines",
        action="store_true",
        help=(
            "Print one LaTeX row per epsilon in format: "
            "& eps & margin_asr & margin_spearman & ce_asr & ce_spearman \\\\"
        ),
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
        help="Where to save evaluation JSON for single mode. Default: <run_root>/evaluation_summary.json",
    )
    return parser.parse_args()


def _extract_eps_and_loss_type(approach):
    if not approach:
        return None, None

    eps_match = re.search(r"__eps-([0-9]+(?:\.[0-9]+)?)__", approach)
    if not eps_match:
        return None, None

    eps_value = float(eps_match.group(1))
    if eps_value.is_integer():
        eps_value = int(eps_value)

    if "fit-negative_cross_entropy_saliency" in approach:
        loss_type = "cross_entropy"
    else:
        loss_type = "margin"

    return eps_value, loss_type


def _format_asr_percent(asr):
    return f"{(float(asr) * 100.0):.2f}"


def _format_spearman(score):
    return f"{float(score):.4f}"


def _latex_bold(text, should_bold):
    if should_bold:
        return f"\\textbf{{{text}}}"
    return text


def _print_latex_rows_from_report_dir(report_dir):
    root = Path(report_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"report directory not found: {root}")

    rows_by_eps = {}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        summary_path = run_dir / "evaluation_summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        eps_value, loss_type = _extract_eps_and_loss_type(summary.get("approach", ""))
        if eps_value is None or loss_type is None:
            continue

        rows_by_eps.setdefault(eps_value, {})
        if loss_type in rows_by_eps[eps_value]:
            # Keep first one in sorted order for deterministic output.
            continue

        rows_by_eps[eps_value][loss_type] = {
            "asr": float(summary.get("attack_success_rate", float("nan"))),
            "spearman": float(summary.get("mean_spearman_adv_vs_clean_saliency", float("nan"))),
        }

    if not rows_by_eps:
        raise ValueError(
            "No evaluation_summary.json found with parseable epsilon in approach"
        )

    for eps_value in sorted(rows_by_eps.keys(), key=float):
        pair = rows_by_eps[eps_value]
        margin = pair.get("margin")
        cross_entropy = pair.get("cross_entropy")
        if margin is None or cross_entropy is None:
            continue

        margin_asr = _format_asr_percent(margin["asr"])
        margin_spearman = _format_spearman(margin["spearman"])
        ce_asr = _format_asr_percent(cross_entropy["asr"])
        ce_spearman = _format_spearman(cross_entropy["spearman"])

        margin_asr_val = float(margin_asr)
        ce_asr_val = float(ce_asr)
        margin_spearman_val = float(margin_spearman)
        ce_spearman_val = float(ce_spearman)

        margin_asr_txt = _latex_bold(margin_asr, margin_asr_val >= ce_asr_val)
        ce_asr_txt = _latex_bold(ce_asr, ce_asr_val >= margin_asr_val)
        margin_spearman_txt = _latex_bold(
            margin_spearman, margin_spearman_val <= ce_spearman_val
        )
        ce_spearman_txt = _latex_bold(
            ce_spearman, ce_spearman_val <= margin_spearman_val
        )

        print(
            f"& {eps_value} & {margin_asr_txt} & {margin_spearman_txt} "
            f"& {ce_asr_txt} & {ce_spearman_txt} \\\\"
        )


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


def _load_gray_image_array(image_path):
    image = Image.open(image_path).convert("L")
    return np.asarray(image, dtype=np.float64)


def _safe_mean(values):
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def _compute_spearman_for_sample(result):
    output_dir = Path(result["output_dir"])
    clean_map_path = output_dir / "clean_map.png"
    adv_map_path = output_dir / "adv_map.png"

    if not clean_map_path.exists() or not adv_map_path.exists():
        return None, "missing_clean_or_adv_map"

    clean_map = _load_gray_image_array(clean_map_path)
    adv_map = _load_gray_image_array(adv_map_path)

    corr = _spearman_rank_corr(clean_map, adv_map)
    return corr, None


def _evaluate_single_report(report_path, max_samples=None):
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"report file not found: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    model_name = report.get("model")

    ok_results = [item for item in report.get("results", []) if item.get("status") == "ok"]
    if max_samples is not None:
        ok_results = ok_results[:max_samples]

    if not ok_results:
        raise ValueError("No successful samples (status='ok') found in report")

    margin_losses = []
    saliency_losses = []
    success_flags = []
    spearman_scores = []
    spearman_failed = 0

    for result in tqdm(ok_results, desc=f"Evaluating {report_path.parent.name}", unit="img"):
        true_label = int(result.get("true_label", result.get("clean_pred", -1)))
        adv_pred = int(result.get("adv_pred", -1))

        margin_losses.append(float(result.get("margin_loss", float("nan"))))
        saliency_losses.append(float(result.get("saliency_loss", float("nan"))))
        success_flags.append(int(adv_pred != true_label and true_label >= 0 and adv_pred >= 0))

        corr, err = _compute_spearman_for_sample(result=result)
        if err is not None:
            spearman_failed += 1
        spearman_scores.append(corr)

    attack_success_rate = float(np.mean(success_flags))
    mean_margin_loss = _safe_mean(margin_losses)
    mean_saliency_loss = _safe_mean(saliency_losses)
    mean_spearman = _safe_mean(spearman_scores)

    return {
        "report_path": str(report_path),
        "model": model_name,
        "approach": report.get("approach"),
        "num_ok_samples_evaluated": len(ok_results),
        "attack_success_rate": attack_success_rate,
        "mean_margin_loss": mean_margin_loss,
        "mean_saliency_loss": mean_saliency_loss,
        "mean_spearman_adv_vs_clean_saliency": mean_spearman,
        "spearman_failed_samples": spearman_failed,
    }


def _save_summary(summary, output_file):
    output_file = Path(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _print_summary(summary, output_file):
    print("=== Evaluation summary ===")
    print(f"report_path: {summary['report_path']}")
    print(f"model: {summary['model']}")
    print(f"approach: {summary['approach']}")
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


def _evaluate_all_runs_in_report_dir(report_dir, max_samples=None):
    report_dir = Path(report_dir)
    if not report_dir.exists() or not report_dir.is_dir():
        raise FileNotFoundError(f"report directory not found: {report_dir}")

    run_dirs = [d for d in sorted(report_dir.iterdir()) if d.is_dir()]
    evaluated = []
    failed = []

    for run_dir in run_dirs:
        report_path = run_dir / "batch_report.json"
        if not report_path.exists():
            failed.append((str(run_dir), "missing batch_report.json"))
            continue

        try:
            summary = _evaluate_single_report(report_path=report_path, max_samples=max_samples)
        except Exception as exc:
            failed.append((str(run_dir), str(exc)))
            continue

        output_file = run_dir / "evaluation_summary.json"
        _save_summary(summary, output_file)
        evaluated.append((run_dir, summary, output_file))

    return evaluated, failed


def main():
    args = parse_args()

    if args.report_dir:
        evaluated, failed = _evaluate_all_runs_in_report_dir(
            report_dir=args.report_dir,
            max_samples=args.max_samples,
        )

        if not evaluated:
            raise ValueError(
                "No batch_report.json found (or all evaluations failed) under --report-dir"
            )

        print("=== Batch directory evaluation ===")
        print(f"report_dir: {args.report_dir}")
        print(f"evaluated_runs: {len(evaluated)}")
        print(f"failed_or_skipped_runs: {len(failed)}")

        for idx, (run_dir, summary, output_file) in enumerate(evaluated, start=1):
            print(
                f"[{idx}] saved: {output_file} "
                f"(ASR={summary['attack_success_rate']:.4f}, "
                f"Spearman={summary['mean_spearman_adv_vs_clean_saliency']:.4f})"
            )

        if failed:
            print("=== Skipped/failed runs ===")
            for run_path, err in failed:
                print(f"- {run_path}: {err}")

        if args.print_latex_lines:
            _print_latex_rows_from_report_dir(args.report_dir)
        return

    if not args.report_path:
        raise ValueError("Please provide --report-path for single evaluation")

    report_path = Path(args.report_path)
    summary = _evaluate_single_report(
        report_path=report_path,
        max_samples=args.max_samples,
    )

    output_file = Path(args.output_file) if args.output_file else report_path.parent / "evaluation_summary.json"
    _save_summary(summary, output_file)
    _print_summary(summary, output_file)


if __name__ == "__main__":
    main()
