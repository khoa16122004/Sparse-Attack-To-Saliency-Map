"""Evaluate clean, adversarial and archived perturbation maps on the clean image."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "core"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

if __package__:
    from .run_score import _normalize_algorithm, _parse_approach, _resolve_run_context
else:
    from run_score import _normalize_algorithm, _parse_approach, _resolve_run_context


RESULT_NAME = "faithfulness.json"
MAP_NAMES = ("clean", "adversarial", "perturbation")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def discover_samples(roots):
    samples = {}
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        for name in ("summarize.json", "summary.json"):
            for path in sorted(root.rglob(name)):
                # The two names can be aliases for the same sample.
                samples[path.parent] = path
    if not samples:
        raise ValueError("No summary.json or summarize.json files found")
    return sorted(samples.values())


def sample_context(path, summary):
    run_dir = next(
        (parent for parent in path.parents if parent.name.startswith("strategy-")),
        None,
    )
    if run_dir is None:
        raise ValueError(f"{path}: no strategy-* experiment directory found")
    meta = _parse_approach(run_dir.name)
    model, algorithm, seed = _resolve_run_context(run_dir, run_dir.parent.parent)
    if meta["eps"] is None or meta["explain_method"] == "unknown":
        raise ValueError(f"{run_dir}: missing eps/exp approach tags")
    fitness = summary.get("fitness_function")
    if fitness is None:
        # _parse_approach only distinguishes the two standard fitness functions.
        fitness = next(
            (token[4:] for token in run_dir.name.split("__") if token.startswith("fit-")),
            "margin_saliency",
        )
    losses = {
        "margin_saliency": "margin_loss",
        "cross_entropy_saliency": "negative_cross_entropy_saliency",
        "negative_cross_entropy_saliency": "negative_cross_entropy_saliency",
    }
    return {
        "model": str(summary.get("model", model)).lower(),
        "explain_method": meta["explain_method"],
        "eps": meta["eps"],
        "loss_type": losses.get(fitness, fitness),
        "algorithm": _normalize_algorithm(summary.get("algorithm", algorithm)),
        "strategy": summary.get("operator_strategy", meta["strategy"]),
        "w_m": meta["w_m"],
        "w_s": meta["w_s"],
        "seed": str(meta["seed"] or seed),
        "approach": run_dir.name,
    }


def validate_map(saliency, image_shape):
    if not isinstance(saliency, torch.Tensor):
        raise ValueError("Expected a saliency tensor")
    saliency = saliency.detach().cpu().float()
    while saliency.ndim > 2 and saliency.shape[0] == 1:
        saliency = saliency[0]
    if tuple(saliency.shape) != tuple(image_shape):
        raise ValueError(f"Map shape {tuple(saliency.shape)} != image shape {image_shape}")
    if not torch.isfinite(saliency).all():
        raise ValueError("Saliency map contains non-finite values")
    return saliency


@torch.no_grad()
def evaluate_map(model, normalize, clean, saliency, target_class, blur_fn,
                 step=224, batch_size=16):
    """Return raw target-class probabilities, without endpoint normalization."""
    if step <= 0 or batch_size <= 0:
        raise ValueError("step and batch_size must be positive")
    if clean.ndim != 4 or clean.shape[:2] != (1, 3):
        raise ValueError("Expected one RGB image with shape (1, 3, H, W)")
    if not torch.isfinite(clean).all() or clean.min() < 0 or clean.max() > 1:
        raise ValueError("Clean image must contain finite RGB values in [0, 1]")
    saliency = validate_map(saliency, clean.shape[-2:])
    count = saliency.numel()
    # Stable row-major ordering makes ties (including constant maps) reproducible.
    order = torch.from_numpy(
        np.argsort(-saliency.numpy().reshape(-1), kind="stable").copy()
    ).to(clean.device)
    pixel_counts = list(range(0, count, step)) + [count]
    fractions = np.asarray(pixel_counts, dtype=np.float64) / count
    blurred = blur_fn(clean)
    if blurred.shape != clean.shape or not torch.isfinite(blurred).all():
        raise ValueError("Blur substrate must be finite and match the image shape")
    result = {"pixel_counts": pixel_counts, "fractions": fractions.tolist()}

    for mode, start, finish in (
        ("ins", blurred.clone(), clean),
        ("del", clean.clone(), torch.zeros_like(clean)),
    ):
        pending = []
        probabilities = []
        previous = 0
        for index, changed in enumerate(pixel_counts):
            coordinates = order[previous:changed]
            start.reshape(1, 3, -1)[:, :, coordinates] = (
                finish.reshape(1, 3, -1)[:, :, coordinates]
            )
            previous = changed
            pending.append(start.clone())
            if len(pending) == batch_size or index == len(pixel_counts) - 1:
                logits = model(normalize(torch.cat(pending, dim=0)))
                if logits.ndim != 2 or not 0 <= target_class < logits.shape[1]:
                    raise ValueError("Target class is outside the model output range")
                probs = logits.softmax(dim=1)[:, target_class]
                if not torch.isfinite(probs).all():
                    raise ValueError("Model returned non-finite probabilities")
                probabilities.extend(probs.cpu().tolist())
                pending.clear()
        values = np.asarray(probabilities, dtype=np.float64)
        # Integrate against the actual fraction: the last step can be shorter.
        area = np.sum((values[:-1] + values[1:]) * np.diff(fractions) / 2)
        result[f"{mode}_curve"] = probabilities
        result[f"{mode}_auc"] = float(area)
    result["imd"] = result["ins_auc"] - result["del_auc"]
    return result


def load_image(path, device):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def evaluate_sample(path, summary, context, model, normalize, explain_fn, blur_fn, args):
    clean = load_image(path.parent / "clean.png", args.device)
    adv = load_image(path.parent / "adv.png", args.device)
    if clean.shape != adv.shape:
        raise ValueError(f"{path.parent}: clean and adv image sizes differ")
    target_class = summary["true_label"]
    if isinstance(target_class, bool) or not isinstance(target_class, int) or target_class < 0:
        raise ValueError(f"{path}: true_label must be a non-negative integer")
    target = torch.tensor([target_class], device=args.device)
    # Explain on each source image, but evaluate every map on the clean image.
    with torch.enable_grad():
        clean_map, _ = explain_fn(model, clean.clone(), normalize, target)
        adv_map, _ = explain_fn(model, adv.clone(), normalize, target)
    density_path = path.parent / "perturbation_density.pt"
    density = torch.load(density_path, map_location="cpu", weights_only=True)
    maps = {
        "clean": clean_map,
        "adversarial": adv_map,
        "perturbation": density["normalized_density"],
    }
    scores = {
        name: evaluate_map(
            model, normalize, clean, saliency, target_class, blur_fn,
            step=args.step, batch_size=args.batch_size,
        )
        for name, saliency in maps.items()
    }
    return {
        "schema_version": 1,
        "status": "ok",
        "sample_dir": str(path.parent.resolve()),
        "context": context,
        "target_class": target_class,
        "protocol": {
            "evaluation_image": "clean",
            "target": "true_label",
            "insertion_baseline": "gaussian_blur_rgb",
            "deletion_baseline": "black_rgb",
            "step": args.step,
            "kernel_size": args.kernel_size,
            "kernel_sigma": args.kernel_sigma,
            "curve_values": "raw_softmax_probability",
            "auc_axis": "fraction_of_pixels",
            "tie_break": "stable_row_major",
            "explanation_source": "recomputed_from_saved_rgb",
            "perturbation_source": "normalized_density_tensor",
        },
        "perturbation_metadata": {
            key: summary.get("perturbation_density", {}).get(key)
            for key in ("grid_size", "grid_value", "sigma")
        },
        "maps": scores,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--step", type=int, default=224, help="Pixels changed per step")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--kernel-sigma", type=float, default=5)
    parser.add_argument("--model", help="Only evaluate this model")
    parser.add_argument("--explain-method", help="Only evaluate this explanation method")
    parser.add_argument("--replace", action="store_true", help="Overwrite existing evaluations")
    args = parser.parse_args()
    if args.step <= 0 or args.batch_size <= 0:
        parser.error("--step and --batch-size must be positive")
    if args.kernel_size <= 0 or args.kernel_size % 2 == 0:
        parser.error("--kernel-size must be a positive odd number")
    if not math.isfinite(args.kernel_sigma) or args.kernel_sigma <= 0:
        parser.error("--kernel-sigma must be finite and positive")
    return args


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; explicitly use --device cpu")
    from util import build_blur_substrate, get_explainable_method, get_torchvision_model
    from RISE.evaluation import gkern

    blur_fn = build_blur_substrate(
        gkern, kernel_size=args.kernel_size, kernel_sigma=args.kernel_sigma
    )
    tasks = []
    for path in discover_samples(args.input_dirs):
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary.get("status") != "ok":
            print(f"[SKIP] {path}: attack status={summary.get('status')}")
            continue
        context = sample_context(path, summary)
        if args.model and context["model"] != args.model.lower():
            continue
        if args.explain_method and context["explain_method"] != args.explain_method:
            continue
        if (path.parent / RESULT_NAME).exists() and not args.replace:
            print(f"[SKIP] {path.parent}: {RESULT_NAME} exists (use --replace)")
            continue
        tasks.append((context["model"], str(path), path, summary, context))
    current_model = None
    evaluated = 0
    for model_name, _, path, summary, context in sorted(tasks):
        if model_name != current_model:
            if current_model is not None:
                del model
            model, _, normalize = get_torchvision_model(model_name, pretrained=True)
            model = model.to(args.device).eval()
            current_model = model_name
        explain_fn = get_explainable_method(context["explain_method"])
        result = evaluate_sample(
            path, summary, context, model, normalize, explain_fn, blur_fn, args
        )
        write_json(path.parent / RESULT_NAME, result)
        evaluated += 1
        metrics = ", ".join(
            f"{name}: IMD={result['maps'][name]['imd']:.4f}" for name in MAP_NAMES
        )
        print(f"[OK] {path.parent} | {metrics}")
    print(f"Evaluated {evaluated} samples. Existing/filtered/non-ok samples were skipped.")


if __name__ == "__main__":
    main()
