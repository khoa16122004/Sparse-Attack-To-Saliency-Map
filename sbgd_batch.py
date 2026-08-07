import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision.utils import save_image
from tqdm.auto import tqdm


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(ROOT_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from explain_method_backprop import get_explainable_method_backprop
from RISE.evaluation import CausalMetric, auc
from spgd import SaliencySparsePGD
from util import build_blur_substrate, get_torchvision_model


DEFAULT_IMAGENET_VAL_ROOT = r"E:\ImageNet1K\imagenet\ImageNet1K\val"
DEFAULT_REMOTE_VAL_ROOT = "/datastore/elo/quanphm/dataset/ImageNet1K/val"


class SoftmaxModel(nn.Module):
    def __init__(self, model, dim=1):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, inputs):
        return self.softmax(self.model(inputs))


def _fmt_num(value):
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}".replace("+", "")
    return str(value)


def _build_approach_tag(args):
    parts = [
        f"wm-{_fmt_num(args.w_margin)}",
        f"ws-{_fmt_num(args.w_saliency)}",
        f"eps-{_fmt_num(args.epsilon)}",
        f"k-{_fmt_num(args.k)}",
        f"iter-{_fmt_num(args.iterations)}",
        f"alpha-{_fmt_num(args.alpha)}",
        f"beta-{_fmt_num(args.beta)}",
        f"exp-{args.explain_method}",
        f"mode-{args.attack_mode}",
    ]
    if args.sparsity_ratio is not None:
        parts.append(f"sr-{_fmt_num(args.sparsity_ratio)}")
    if args.seed is not None:
        parts.append(f"seed-{args.seed}")
    return "__".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch runner for saliency-based sparse gradient attack")
    parser.add_argument("--selection-file", type=str, default=None, help="Path to *_selection.json")
    parser.add_argument("--model-name", type=str, default=None, help="Torchvision model name")
    parser.add_argument("--num_sample", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--imagenet-val-root", type=str, default=DEFAULT_IMAGENET_VAL_ROOT)
    parser.add_argument("--replace-from-root", type=str, default=DEFAULT_REMOTE_VAL_ROOT)
    parser.add_argument("--output-root", type=str, default="sbgd_batch_outputs")
    parser.add_argument("--replace", action="store_true", help="Overwrite existing outputs")

    parser.add_argument(
        "--explain-method",
        type=str,
        default="input_gradient",
        choices=["simple_gradient", "integrated_gradients", "input_gradient", "raw_attention", "attention_grad"],
    )
    parser.add_argument("--label", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None, help="L-inf budget in [0,1] scale")
    parser.add_argument("--eps", type=float, default=8.0, help="Pixel-scale budget (converted to epsilon=eps/255)")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--sparsity-ratio", type=float, default=None)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=1.0 / 255.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--attack-mode", type=str, default="pixel", choices=["pixel", "feature"])
    parser.add_argument("--w-margin", type=float, default=1.0)
    parser.add_argument("--w-saliency", type=float, default=1.0)
    parser.add_argument("--disable-softplus-surrogate", action="store_true")
    parser.add_argument("--softplus-beta", type=float, default=10.0)
    parser.add_argument("--disable-fixed-mask-location", action="store_true")
    parser.add_argument("--zero-grad-patience", type=int, default=3)
    parser.add_argument("--zero-grad-jitter", type=float, default=1e-2)
    parser.add_argument("--debug-grad", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--step", type=int, default=224)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--kernel-sigma", type=int, default=5)
    parser.add_argument("--save-process", action="store_true")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])

    return parser.parse_args()


def _save_saliency_map(saliency_map, output_path):
    map_2d = saliency_map.detach().float().cpu().squeeze()
    map_min = map_2d.min()
    map_max = map_2d.max()
    den = (map_max - map_min).item()
    if den > 1e-12:
        map_2d = (map_2d - map_min) / (map_max - map_min)
    else:
        map_2d = torch.zeros_like(map_2d)

    r = torch.clamp(3.0 * map_2d, 0.0, 1.0)
    g = torch.clamp(3.0 * map_2d - 1.0, 0.0, 1.0)
    b = torch.clamp(3.0 * map_2d - 2.0, 0.0, 1.0)
    rgb = (torch.stack([r, g, b], dim=-1) * 255.0).clamp(0, 255).byte().numpy()

    from PIL import Image as PILImage

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    PILImage.fromarray(rgb, mode="RGB").save(output_path)


def _soft_iou(s1, s2, eps=1e-12):
    s1 = s1.flatten(start_dim=1)
    s2 = s2.flatten(start_dim=1)
    inter = torch.minimum(s1, s2).sum(dim=1)
    union = torch.maximum(s1, s2).sum(dim=1)
    return inter / (union + eps)


def _mse_per_sample(s1, s2):
    s1 = s1.flatten(start_dim=1)
    s2 = s2.flatten(start_dim=1)
    if s1.size(0) == 1 and s2.size(0) > 1:
        s1 = s1.expand(s2.size(0), -1)
    return ((s1 - s2) ** 2).mean(dim=1)


def _margin_loss(logits, y_true):
    true_logits = logits.gather(1, y_true.view(-1, 1)).squeeze(1)
    others = logits.clone()
    others.scatter_(1, y_true.view(-1, 1), float("-inf"))
    other_logits = others.max(dim=1).values
    return -(true_logits - other_logits)


def _compute_causal_metrics(model, normalize, clean_batch, adv_batch, clean_saliency_map, adv_saliency_map, output_dir, args):
    metric_model = SoftmaxModel(model)

    blur_fn = build_blur_substrate(args.kernel_size, args.kernel_sigma)
    insertion = CausalMetric(metric_model, "ins", args.step, substrate_fn=blur_fn)
    deletion = CausalMetric(metric_model, "del", args.step, substrate_fn=lambda x: torch.zeros_like(x))

    clean_del_steps_dir = output_dir / "clean_deletion_steps"
    clean_ins_steps_dir = output_dir / "clean_insertion_steps"
    adv_del_steps_dir = output_dir / "adv_deletion_steps"
    adv_ins_steps_dir = output_dir / "adv_insertion_steps"

    if args.save_process:
        clean_del_steps_dir.mkdir(parents=True, exist_ok=True)
        clean_ins_steps_dir.mkdir(parents=True, exist_ok=True)
        adv_del_steps_dir.mkdir(parents=True, exist_ok=True)
        adv_ins_steps_dir.mkdir(parents=True, exist_ok=True)

    clean_scores_del = deletion.single_run(
        normalize(clean_batch).cpu().detach(),
        clean_saliency_map.cpu().detach().numpy(),
        verbose=args.verbose,
        save_to=str(clean_del_steps_dir) if args.save_process else None,
    )
    clean_scores_ins = insertion.single_run(
        normalize(clean_batch).cpu().detach(),
        clean_saliency_map.cpu().detach().numpy(),
        verbose=args.verbose,
        save_to=str(clean_ins_steps_dir) if args.save_process else None,
    )

    adv_scores_del = deletion.single_run(
        normalize(adv_batch).cpu().detach(),
        clean_saliency_map.cpu().detach().numpy(),
        verbose=args.verbose,
        save_to=str(adv_del_steps_dir) if args.save_process else None,
    )
    adv_scores_ins = insertion.single_run(
        normalize(adv_batch).cpu().detach(),
        adv_saliency_map[0].cpu().detach().numpy(),
        verbose=args.verbose,
        save_to=str(adv_ins_steps_dir) if args.save_process else None,
    )

    return {
        "clean_del_auc": float(auc(clean_scores_del)),
        "clean_ins_auc": float(auc(clean_scores_ins)),
        "adv_del_auc": float(auc(adv_scores_del)),
        "adv_ins_auc": float(auc(adv_scores_ins)),
        "clean_del_scores": [float(v) for v in clean_scores_del],
        "clean_ins_scores": [float(v) for v in clean_scores_ins],
        "adv_del_scores": [float(v) for v in adv_scores_del],
        "adv_ins_scores": [float(v) for v in adv_scores_ins],
    }


def _infer_model_name(selection_path):
    stem = Path(selection_path).stem
    suffix = "_selection"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _resolve_image_path(raw_path, class_name, imagenet_val_root, replace_from_root):
    image_path = Path(raw_path)
    if image_path.exists():
        return image_path

    normalized_raw = str(raw_path).replace("\\", "/")
    normalized_old_root = replace_from_root.replace("\\", "/").rstrip("/")

    if normalized_raw.startswith(normalized_old_root + "/"):
        suffix = normalized_raw[len(normalized_old_root) + 1 :]
        candidate = Path(imagenet_val_root) / Path(suffix)
        if candidate.exists():
            return candidate

    image_name = Path(raw_path).name
    return Path(imagenet_val_root) / class_name / image_name


def _run_one(model, spatial, normalize, explain_fn, args, image_path, output_dir, sample_seed):
    if sample_seed is not None:
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)

    device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    x = spatial(image).to(device).unsqueeze(0)

    with torch.no_grad():
        clean_logits = model(normalize(x))
        clean_pred = clean_logits.argmax(dim=1)

    y_true = clean_pred if args.label is None else torch.tensor([args.label], device=device, dtype=torch.long)

    attacker = SaliencySparsePGD(
        model=model,
        normalize=normalize,
        explain_method=explain_fn,
        epsilon=args.epsilon,
        k=args.k,
        sparsity_ratio=args.sparsity_ratio,
        tau=args.tau,
        t=args.iterations,
        alpha=args.alpha,
        beta=args.beta,
        attack_mode=args.attack_mode,
        w_margin=args.w_margin,
        w_saliency=args.w_saliency,
        fixed_mask_location=not args.disable_fixed_mask_location,
        use_softplus_surrogate=not args.disable_softplus_surrogate,
        softplus_beta=args.softplus_beta,
        zero_grad_patience=args.zero_grad_patience,
        zero_grad_jitter=args.zero_grad_jitter,
        debug_grad=args.debug_grad,
    )

    x_adv, history = attacker.attack(x, y_true, return_history=True)

    adv_path = output_dir / "adv.png"
    clean_path = output_dir / "clean.png"
    clean_map_path = output_dir / "clean_map.png"
    adv_map_path = output_dir / "adv_map.png"
    summary_path = output_dir / "summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(x[0].detach().cpu(), str(clean_path))
    save_image(x_adv[0].detach().cpu(), str(adv_path))

    clean_saliency, _ = explain_fn(model, x, normalize, target_class=y_true)
    adv_saliency, adv_logits = explain_fn(model, x_adv, normalize, target_class=y_true)
    adv_pred = adv_logits.argmax(dim=1)

    _save_saliency_map(clean_saliency[0], str(clean_map_path))
    _save_saliency_map(adv_saliency[0], str(adv_map_path))

    causal = _compute_causal_metrics(
        model=model,
        normalize=normalize,
        clean_batch=x,
        adv_batch=x_adv.detach(),
        clean_saliency_map=clean_saliency,
        adv_saliency_map=adv_saliency,
        output_dir=output_dir,
        args=args,
    )

    margin = _margin_loss(adv_logits, y_true).mean().item()
    saliency_mse = _mse_per_sample(clean_saliency, adv_saliency).mean().item()

    payload = {
        "image": str(image_path),
        "label_used": int(y_true[0].item()),
        "clean_pred": int(clean_pred[0].item()),
        "adv_pred": int(adv_pred[0].item()),
        "margin_loss": float(margin),
        "saliency_mse": float(saliency_mse),
        "objective": float(args.w_margin * margin + args.w_saliency * saliency_mse),
        "history": history,
        "output_adv": str(adv_path),
        "output_clean": str(clean_path),
        "causual": {
            "del": causal["adv_del_scores"],
            "ins": causal["adv_ins_scores"],
            "clean_del": causal["clean_del_scores"],
            "clean_ins": causal["clean_ins_scores"],
            "auc": {
                "del": causal["adv_del_auc"],
                "ins": causal["adv_ins_auc"],
                "clean_del": causal["clean_del_auc"],
                "clean_ins": causal["clean_ins_auc"],
            },
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def run(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.num_sample is not None and args.num_sample < 0:
        raise ValueError("--num_sample must be >= 0")

    if args.selection_file is None:
        if not args.model_name:
            raise ValueError("Provide --model-name or --selection-file")
        selection_file = Path("model_evaluation_results") / f"{args.model_name}_selection_random50.json"
    else:
        selection_file = Path(args.selection_file)

    if not selection_file.exists():
        raise FileNotFoundError(f"Selection file not found: {selection_file}")

    args.epsilon = float(args.epsilon) if args.epsilon is not None else float(args.eps) / 255.0

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU")
        args.device = "cpu"

    with open(selection_file, "r", encoding="utf-8") as f:
        selection = json.load(f)
    if not isinstance(selection, dict):
        raise ValueError("selection-file must be a JSON object mapping class to image path")

    model_name = args.model_name if args.model_name is not None else _infer_model_name(str(selection_file))
    print(f"[INFO] selection_file={selection_file}")
    print(f"[INFO] model={model_name}")
    print(f"[INFO] device={args.device}")
    print(f"[INFO] seed={args.seed}")
    print(f"[INFO] epsilon={args.epsilon:.8f} (eps={args.eps})")
    model, spatial, normalize = get_torchvision_model(model_name, pretrained=True)
    model = model.to(torch.device(args.device))
    model.eval()

    explain_fn = get_explainable_method_backprop(args.explain_method)

    approach_tag = _build_approach_tag(args)
    output_root = Path(args.output_root) / model_name
    run_root = output_root / approach_tag
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] run_dir={run_root}")

    items = list(selection.items())
    if args.num_sample is not None:
        items = items[: args.num_sample]

    results = []
    iterator = tqdm(items, total=len(items), desc=f"sbgd-batch:{model_name}")
    for idx, (class_name, raw_path) in enumerate(iterator):
        image_path = _resolve_image_path(raw_path, class_name, args.imagenet_val_root, args.replace_from_root)
        sample_name = f"{idx:04d}_{class_name}"
        sample_dir = run_root / sample_name
        summary_file = sample_dir / "summary.json"

        if summary_file.exists() and not args.replace:
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                results.append(cached)
                continue
            except Exception:
                pass

        if not image_path.exists():
            results.append(
                {
                    "image": str(raw_path),
                    "error": f"Image not found: {image_path}",
                    "class_name": class_name,
                }
            )
            continue

        sample_seed = None if args.seed is None else args.seed + idx
        out = _run_one(
            model=model,
            spatial=spatial,
            normalize=normalize,
            explain_fn=explain_fn,
            args=args,
            image_path=image_path,
            output_dir=sample_dir,
            sample_seed=sample_seed,
        )
        out["class_name"] = class_name
        results.append(out)

    report = {
        "selection_file": str(selection_file),
        "model": model_name,
        "approach_tag": approach_tag,
        "num_requested": args.num_sample,
        "num_processed": len(results),
        "output_root": str(run_root),
        "results": results,
    }

    summary_path = run_root / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Processed {len(results)} samples")
    print(f"Saved batch summary: {summary_path}")


if __name__ == "__main__":
    run(parse_args())
