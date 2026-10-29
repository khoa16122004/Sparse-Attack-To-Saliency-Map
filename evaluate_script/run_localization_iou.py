"""CPU-only localization IoU from saved Clean and perturbation-density maps."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluate_latex_generate.generate_tables import add_seed, cell_key, summarize_cells
from evaluate_script.generate_imagenet_masks import sha256
from evaluate_script.run_aggregate_faithfulness import LABELS, approach_name
from evaluate_script.run_evaluate_faithfulness import discover_samples, sample_context

APPROACHES = ("clean", "ga_0", "ga_0.5", "ga_1", "nsgaii")


def top_k_mask(saliency, percent):
    saliency = np.asarray(saliency, dtype=np.float64)
    if saliency.ndim != 2 or not saliency.size or not np.isfinite(saliency).all():
        raise ValueError("Saliency must be a nonempty finite 2D array")
    if not math.isfinite(percent) or not 0 < percent <= 100:
        raise ValueError("Top-k percentage must be in (0, 100]")
    count = math.ceil(saliency.size * percent / 100)
    order = np.argsort(-saliency.reshape(-1), kind="stable")
    result = np.zeros(saliency.size, dtype=bool)
    result[order[:count]] = True
    return result.reshape(saliency.shape)


def binary_iou(prediction, target):
    if prediction.shape != target.shape or prediction.dtype != bool or target.dtype != bool:
        raise ValueError("IoU requires boolean masks with identical shapes")
    union = np.count_nonzero(prediction | target)
    if union == 0:
        raise ValueError("IoU is undefined for two empty masks")
    return float(np.count_nonzero(prediction & target) / union)


def load_clean_map(directory):
    tensor_path = directory / "clean_map.pt"
    if tensor_path.is_file():
        tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"{tensor_path}: expected a 2D saliency tensor")
        return tensor.float().numpy(), "raw_tensor"
    path = directory / "clean_map.png"
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"{path}: expected the project's RGB saliency encoding")
        rgb = np.asarray(image, dtype=np.float64)
    # Invert the monotone 3-channel ramp used by run_batch._save_saliency_map.
    values = rgb.sum(axis=2) / (3 * 255)
    expected = np.stack([
        np.clip(3 * values, 0, 1), np.clip(3 * values - 1, 0, 1),
        np.clip(3 * values - 2, 0, 1),
    ], axis=-1) * 255
    if not np.allclose(rgb, expected, rtol=0, atol=1.01):
        raise ValueError(f"{path}: not a supported saliency heatmap (do not use an overlay)")
    return values, "inverted_quantized_rgb"


def align_target(source_path, mask_path, clean_path, model_name):
    from torchvision.models import get_model_weights
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    # Read weights metadata only: no model construction or checkpoint download.
    weights = get_model_weights(model_name).DEFAULT.transforms()
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        original_size = image.size
        expected = TF.center_crop(TF.resize(image, weights.resize_size), weights.crop_size)
    with Image.open(clean_path) as image:
        actual = np.asarray(image.convert("RGB"), dtype=np.int16)
    expected_array = np.asarray(expected, dtype=np.int16)
    if actual.shape != expected_array.shape or np.max(np.abs(actual - expected_array)) > 1:
        raise ValueError(
            f"{clean_path}: saved clean image does not match {source_path} with "
            f"{model_name}'s Resize/CenterCrop; refusing misaligned IoU"
        )
    with Image.open(mask_path) as mask:
        if mask.mode != "L" or mask.size != original_size:
            raise ValueError(f"{mask_path}: expected original-resolution grayscale mask")
        values = np.asarray(mask)
        if not np.isin(values, [0, 255]).all():
            raise ValueError(f"{mask_path}: mask is not binary 0/255")
        mask = TF.resize(mask, weights.resize_size, interpolation=InterpolationMode.NEAREST)
        mask = TF.center_crop(mask, weights.crop_size)
        return np.asarray(mask) == 255


def latex_escape(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def save_reports(records, output_dir, precision):
    groups = defaultdict(list)
    for record in records:
        context = record["context"]
        config = {
            "eps": context["eps"], "loss_type": context["loss_type"],
            "strategy": context["strategy"], "protocol": record["protocol"],
            "perturbation_metadata": record["perturbation_metadata"],
        }
        groups[json.dumps(config, sort_keys=True, allow_nan=False)].append(record)
    for config, entries in sorted(groups.items()):
        buckets = defaultdict(dict)
        for record in entries:
            context = record["context"]
            identity = (str(context["seed"]), *record["identity"])
            for name, value in (("clean", record["clean_iou"]),
                                (approach_name(context), record["perturbation_iou"])):
                key = cell_key(context["model"], context["explain_method"], name)
                previous = buckets[key].get(identity)
                fingerprint = (record["clean_rgb_sha256"], record["gt_mask_sha256"],
                               record["clean_binary_sha256"])
                if previous is not None:
                    if name != "clean":
                        raise ValueError(f"Duplicate IoU evaluation for {key}, {identity}")
                    if previous != (value, fingerprint):
                        raise ValueError(f"Inconsistent Clean/GT across approaches for {key}, {identity}")
                buckets[key][identity] = (value, fingerprint)
        cells = defaultdict(dict)
        sample_rows = []
        for key, samples in sorted(buckets.items()):
            by_seed = defaultdict(list)
            for (seed, class_id, stem), (value, _fingerprint) in sorted(samples.items()):
                by_seed[seed].append(((class_id, stem), value))
                sample_rows.append(dict(
                    model=key[0], explain_method=key[1], approach=key[2],
                    seed=seed, class_id=class_id, image_stem=stem, iou=value,
                ))
            for seed, values in by_seed.items():
                add_seed(cells, key, seed, {"iou": float(np.mean([v for _, v in values]))},
                         [list(identity) for identity, _ in values])
        summary = summarize_cells(cells, ("iou",))
        folder = output_dir / ("localization_" + hashlib.sha256(config.encode()).hexdigest()[:12])
        folder.mkdir(parents=True, exist_ok=True)
        report = {
            "configuration": json.loads(config), "averaging": "equal_weight_seed_means",
            "std": "sample_std_ddof_1", "cells": [
                {"model": key[0], "explain_method": key[1], "approach": key[2], **stats}
                for key, stats in sorted(summary.items())
            ], "records": entries,
        }
        (folder / "statistics.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        with (folder / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
            writer.writeheader()
            writer.writerows(sample_rows)
        percent = entries[0]["protocol"]["top_k_percent"]
        lines = [
            r"\begin{table*}[t]", r"\centering",
            rf"\caption{{Localization IoU against SAM3 pseudo-GT; top {percent:g}\% pixels. "
            r"Samples are averaged within each seed, then mean $\pm$ sample SD across seeds. "
            r"Single-seed SD and missing cells are --; bold marks the best mean.}",
            r"\begin{tabular}{lllcc}", r"\hline",
            r"Model & Explainer & Approach & IoU $\uparrow$ & Seeds \\",
            r"\hline",
        ]
        for model, method in sorted({key[:2] for key in summary}):
            best = max(stats["iou"]["mean"] for key, stats in summary.items() if key[:2] == (model, method))
            for name in APPROACHES:
                stats = summary.get((model, method, name))
                value, count = "--", "0"
                if stats:
                    stat = stats["iou"]
                    sd = "--" if stat["std"] is None else f"{stat['std']:.{precision}f}"
                    value = f"{stat['mean']:.{precision}f}" + r" \pm " + sd
                    if stat["mean"] == best:
                        value = r"\mathbf{" + value + "}"
                    value = "$" + value + "$"
                    count = str(stat["n_seeds"])
                lines.append(
                    f"{latex_escape(model)} & {latex_escape(method)} & {LABELS[name]} & "
                    f"{value} & {count}" + r" \\"
                )
            lines.append(r"\hline")
        lines.extend([r"\end{tabular}", r"\end{table*}", ""])
        (folder / "table.tex").write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] {folder}: samples.csv, statistics.json, table.tex")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--imagenet-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, default=Path("imagenet_mask"))
    parser.add_argument("--output-dir", type=Path, default=Path("result_latexs"))
    parser.add_argument("--records-output", type=Path,
                        help="Write exact per-sample records for the combined faithfulness table")
    parser.add_argument("--top-k-percent", type=float, default=80)
    parser.add_argument("--empty-gt", choices=("error", "zero"), default="error",
                        help="Default rejects empty SAM/cropped masks; zero explicitly scores them as 0")
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--eps", nargs="+", type=int)
    parser.add_argument("--loss-type")
    parser.add_argument("--strategy")
    args = parser.parse_args(argv)
    if not math.isfinite(args.top_k_percent) or not 0 < args.top_k_percent <= 100:
        parser.error("--top-k-percent must be in (0, 100]")
    if not 0 <= args.precision <= 12:
        parser.error("--precision must be in [0, 12]")
    if not args.imagenet_root.is_dir():
        raise NotADirectoryError(args.imagenet_root)
    manifest_path = args.mask_root / "mask_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError(f"{manifest_path}: expected a completed schema-version-1 mask generation")
    index = {}
    for relative, item in manifest["images"].items():
        relative = Path(relative)
        key = (relative.parts[0], relative.stem)
        if key in index:
            raise ValueError(f"Ambiguous class/image stem in mask manifest: {key}")
        index[key] = (relative, item)
    records = []
    manifest_hash = sha256(manifest_path)
    for path in discover_samples(args.input_dirs):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "ok":
            print(f"[SKIP] {path}: attack status={summary.get('status')}")
            continue
        context = sample_context(path, summary)
        if ((args.eps is not None and context["eps"] not in args.eps)
                or (args.loss_type and context["loss_type"] != args.loss_type)
                or (args.strategy and context["strategy"] != args.strategy)):
            continue
        if str(context["seed"]) in ("", "None", "default", "unknown"):
            raise ValueError(f"{path}: explicit seed required")
        identity = (path.parent.parent.name, path.parent.name)
        relative, item = index[identity]
        source_path = args.imagenet_root / relative
        mask_path = args.mask_root / item["mask"]
        if sha256(source_path) != item["source_sha256"] or sha256(mask_path) != item["mask_sha256"]:
            raise ValueError(f"{path}: source image or GT mask changed; regenerate masks")
        target = align_target(source_path, mask_path, path.parent / "clean.png", context["model"])
        if not target.any():
            if args.empty_gt == "error":
                raise ValueError(f"{mask_path}: empty GT after crop; review prompt or explicitly use --empty-gt zero")
            print(f"[EMPTY GT -> IoU=0] {path.parent}")
        clean_map, clean_source = load_clean_map(path.parent)
        archive_path = path.parent / "perturbation_density.pt"
        archive = torch.load(archive_path, map_location="cpu", weights_only=True)
        density = archive["smoothed_density"]
        if not isinstance(density, torch.Tensor) or density.ndim != 2:
            raise ValueError(f"{archive_path}: expected 2D smoothed_density tensor")
        clean_binary = top_k_mask(clean_map, args.top_k_percent)
        perturbation_binary = top_k_mask(density.float().numpy(), args.top_k_percent)
        records.append({
            "context": context, "identity": list(identity), "sample_dir": str(path.parent),
            "clean_iou": binary_iou(clean_binary, target),
            "perturbation_iou": binary_iou(perturbation_binary, target),
            "clean_rgb_sha256": sha256(path.parent / "clean.png"),
            "gt_mask_sha256": item["mask_sha256"],
            "clean_binary_sha256": hashlib.sha256(clean_binary.tobytes()).hexdigest(),
            "gt_foreground_pixels": int(target.sum()),
            "protocol": {
                "top_k_percent": args.top_k_percent, "pixel_count": "ceil(H*W*k/100)",
                "tie_break": "stable_row_major", "empty_gt": args.empty_gt,
                "gt_manifest_sha256": manifest_hash, "gt": "sam3_pseudo_gt",
                "alignment": "model_default_resize_center_crop_nearest_mask",
                "clean_source": clean_source, "perturbation_source": "smoothed_density_tensor",
            },
            "perturbation_metadata": {
                key: summary.get("perturbation_density", {}).get(key)
                for key in ("grid_size", "grid_value", "sigma")
            },
        })
    if not records:
        raise ValueError("No matching successful samples for localization IoU")
    save_reports(records, args.output_dir, args.precision)
    if args.records_output:
        args.records_output.parent.mkdir(parents=True, exist_ok=True)
        args.records_output.write_text(
            json.dumps({"schema_version": 1, "records": records}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
