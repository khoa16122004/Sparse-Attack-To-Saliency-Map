"""Re-evaluate saved attacks and write one metrics file per sample, without aggregation."""

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm.auto import tqdm

from evaluate_script.generate_imagenet_masks import sha256
from evaluate_script.run_evaluate_faithfulness import (
    discover_samples, evaluate_sample, sample_context, write_json,
)
from evaluate_script.run_localization_iou import (
    align_target, binary_iou, load_clean_map, threshold_mask, top_k_mask,
)
from evaluate_script.run_score import (
    _load_gray_image_array, _read_history_from_text, _spearman_rank_corr,
)


RESULT_NAME = "evaluation.json"


def load_mask_index(mask_root):
    manifest_path = mask_root / "mask_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError(f"{manifest_path}: expected a complete schema-version-1 manifest")
    index = {}
    for name, item in manifest["images"].items():
        relative = Path(name)
        if len(relative.parts) < 2:
            raise ValueError(f"{manifest_path}: invalid image path {name!r}")
        key = (relative.parts[0], relative.stem)
        if key in index:
            raise ValueError(f"{manifest_path}: ambiguous class/image stem {key}")
        index[key] = (relative, item)
    return index, sha256(manifest_path)


def attack_metrics(path, summary):
    true_label = summary.get("true_label", summary.get("clean_pred"))
    adv_pred = summary.get("adv_pred")
    for name, value in (("true_label", true_label), ("adv_pred", adv_pred)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{path}: {name} must be a non-negative integer")
    clean_map = _load_gray_image_array(path.parent / "clean_map.png")
    adv_map = _load_gray_image_array(path.parent / "adv_map.png")
    if clean_map.shape != adv_map.shape:
        raise ValueError(f"{path}: clean and adversarial map shapes differ")
    correlation = _spearman_rank_corr(clean_map, adv_map)
    if not math.isfinite(correlation):
        correlation = None
    history_margin = summary.get("history_margin")
    history_saliency = summary.get("history_saliency")
    if not history_margin or not history_saliency:
        margin, saliency = _read_history_from_text(path.parent / "history_scores.txt")
        if not margin and not saliency:
            margin, saliency = _read_history_from_text(path.parent / "history.txt")
        history_margin = history_margin or margin
        history_saliency = history_saliency or saliency
    return {
        "true_label": true_label,
        "clean_pred": summary.get("clean_pred"),
        "adv_pred": adv_pred,
        "success": adv_pred != true_label,
        "first_success_iteration": summary.get("first_success_iteration"),
        "spearman": correlation,
        "spearman_error": "constant_or_too_small_map" if correlation is None else None,
        "history_margin": history_margin,
        "history_saliency": history_saliency,
        "history_attack_objective": summary.get("history_attack_objective"),
        "history_objective": summary.get("history_objective"),
        "history_ce": summary.get("history_ce"),
        "history_cross_entropy": summary.get("history_cross_entropy"),
        "l0_distance": summary.get("l0_distance"),
    }


def localization_metrics(path, context, args, index, manifest_hash):
    identity = (path.parent.parent.name, path.parent.name)
    if identity not in index:
        raise ValueError(f"{path}: no GT mask for {identity} in mask manifest")
    relative, item = index[identity]
    source_path = args.imagenet_root / relative
    mask_path = args.mask_root / item["mask"]
    if sha256(source_path) != item["source_sha256"] or sha256(mask_path) != item["mask_sha256"]:
        raise ValueError(f"{path}: source image or GT mask changed; regenerate masks")
    target = align_target(source_path, mask_path, path.parent / "clean.png", context["model"])
    if not target.any() and args.empty_gt == "error":
        raise ValueError(f"{mask_path}: empty GT after crop; use --empty-gt zero to score it as 0")

    clean_map, clean_source = load_clean_map(path.parent)
    density_path = path.parent / "perturbation_density.pt"
    archive = torch.load(density_path, map_location="cpu", weights_only=True)
    density = archive["smoothed_density"]
    if not isinstance(density, torch.Tensor) or density.ndim != 2:
        raise ValueError(f"{density_path}: expected 2D smoothed_density tensor")
    if args.top_k_percent is not None:
        mask_fn = lambda values: top_k_mask(values, args.top_k_percent)
        protocol = {
            "top_k_percent": args.top_k_percent, "pixel_count": "ceil(H*W*k/100)",
            "tie_break": "stable_row_major",
        }
    else:
        mask_fn = lambda values: threshold_mask(values, args.threshold)
        protocol = {
            "threshold": args.threshold, "normalization": "per_map_min_max",
            "comparison": ">=", "constant_map": "normalized_to_zero",
        }
    clean_binary = mask_fn(clean_map)
    density_binary = mask_fn(density.float().numpy())
    if clean_binary.shape != target.shape or density_binary.shape != target.shape:
        raise ValueError(f"{path}: saliency maps and aligned GT mask shapes differ")
    return {
        "clean_iou": binary_iou(clean_binary, target) if target.any() else 0.0,
        "perturbation_iou": binary_iou(density_binary, target) if target.any() else 0.0,
        "gt_foreground_pixels": int(target.sum()),
        "source_sha256": item["source_sha256"],
        "gt_mask_sha256": item["mask_sha256"],
        "protocol": {
            **protocol, "empty_gt": args.empty_gt, "gt": "sam3_pseudo_gt",
            "gt_manifest_sha256": manifest_hash,
            "alignment": "model_default_resize_center_crop_nearest_mask",
            "clean_source": clean_source, "perturbation_source": "smoothed_density_tensor",
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--imagenet-root", type=Path, required=True,
                        help="Original images used to generate the GT mask manifest")
    parser.add_argument("--mask-root", type=Path, required=True,
                        help="SAM3 pseudo-GT masks and mask_manifest.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--step", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--kernel-sigma", type=float, default=5)
    masks = parser.add_mutually_exclusive_group()
    masks.add_argument("--threshold", type=float, default=0.5)
    masks.add_argument("--top-k-percent", type=float)
    parser.add_argument("--empty-gt", choices=("error", "zero"), default="error")
    parser.add_argument("--model", help="Only evaluate this model")
    parser.add_argument("--explain-method", help="Only evaluate this explanation method")
    parser.add_argument("--replace", action="store_true", help="Overwrite existing evaluation.json")
    args = parser.parse_args(argv)
    if args.step <= 0 or args.batch_size <= 0:
        parser.error("--step and --batch-size must be positive")
    if args.kernel_size <= 0 or args.kernel_size % 2 == 0:
        parser.error("--kernel-size must be a positive odd number")
    if not math.isfinite(args.kernel_sigma) or args.kernel_sigma <= 0:
        parser.error("--kernel-sigma must be finite and positive")
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    if args.top_k_percent is not None and (
        not math.isfinite(args.top_k_percent) or not 0 < args.top_k_percent <= 100
    ):
        parser.error("--top-k-percent must be in (0, 100]")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; explicitly use --device cpu")
    if not args.imagenet_root.is_dir():
        raise NotADirectoryError(args.imagenet_root)
    index, manifest_hash = load_mask_index(args.mask_root)
    samples = []
    for path in tqdm(discover_samples(args.input_dirs), desc="Read summaries", unit="sample"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "ok":
            tqdm.write(f"[SKIP] {path}: attack status={summary.get('status')}")
            continue
        context = sample_context(path, summary)
        if args.model and context["model"] != args.model.lower():
            continue
        if args.explain_method and context["explain_method"] != args.explain_method:
            continue
        if (path.parent / RESULT_NAME).exists() and not args.replace:
            raise FileExistsError(f"{path.parent / RESULT_NAME}: use --replace to recompute")
        samples.append((path, summary, context))
    if not samples:
        raise ValueError("No matching successful samples to evaluate")

    from RISE.evaluation import gkern
    from util import build_blur_substrate, get_explainable_method, get_torchvision_model

    blur = build_blur_substrate(gkern, kernel_size=args.kernel_size, kernel_sigma=args.kernel_sigma)
    current_model = None
    samples.sort(key=lambda sample: (sample[2]["model"], str(sample[0])))
    for path, summary, context in tqdm(samples, desc="Evaluate", unit="sample"):
        attack = attack_metrics(path, summary)
        localization = localization_metrics(path, context, args, index, manifest_hash)
        if context["model"] != current_model:
            if current_model is not None:
                del model
            model, _, normalize = get_torchvision_model(context["model"], pretrained=True)
            model = model.to(args.device).eval()
            current_model = context["model"]
        explain_fn = get_explainable_method(context["explain_method"])
        faithfulness = evaluate_sample(
            path, {**summary, "true_label": attack["true_label"]},
            context, model, normalize, explain_fn, blur, args,
        )
        result = {
            "schema_version": 1,
            "status": "ok",
            "sample_dir": str(path.parent.resolve()),
            "summary_file": str(path.resolve()),
            "context": context,
            "attack": attack,
            "faithfulness": {
                key: faithfulness[key] for key in (
                    "clean_image_sha256", "target_class", "protocol",
                    "perturbation_metadata", "maps",
                )
            },
            "localization": localization,
        }
        write_json(path.parent / RESULT_NAME, result)
        tqdm.write(f"[OK] {path.parent}")
    print(f"Evaluated {len(samples)} samples; no aggregation was performed.")


if __name__ == "__main__":
    main()
