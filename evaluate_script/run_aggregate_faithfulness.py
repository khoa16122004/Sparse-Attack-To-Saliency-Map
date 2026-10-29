"""Aggregate faithfulness metrics and curves for Clean, GASS and NSGA-II maps.

Paper-ready LaTeX tables are rendered by evaluate_latex_generate.generate_tables
from per-sample faithfulness.json files using the fixed faithfulness template.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


MAP_NAMES = ("clean", "perturbation")
METRICS = ("ins_auc", "del_auc", "imd")
APPROACHES = ("clean", "ga_0", "ga_0.5", "ga_1", "nsgaii")
LABELS = {
    "clean": "Clean", "ga_0": r"GASS ($\lambda=0.0$)",
    "ga_0.5": r"GASS ($\lambda=0.5$)", "ga_1": r"GASS ($\lambda=1.0$)",
    "nsgaii": "Ours",
}
CURVE_FIELDS = ("fractions", "ins_curve", "del_curve")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def model_family(model):
    return "ViT" if model.startswith(("vit", "swin", "maxvit")) else "CNN"


def validate_record(payload, path):
    if payload.get("schema_version") != 1 or payload.get("status") != "ok":
        raise ValueError(f"{path}: not a successful schema-version-1 evaluation")
    context = payload["context"]
    required = ("model", "explain_method", "eps", "loss_type", "algorithm",
                "strategy", "w_m", "w_s", "seed")
    if any(key not in context for key in required):
        raise ValueError(f"{path}: incomplete experiment context")
    if payload["protocol"]["curve_values"] != "raw_softmax_probability":
        raise ValueError(f"{path}: expected raw probabilities")
    if payload["protocol"]["evaluation_image"] != "clean":
        raise ValueError(f"{path}: all maps must be evaluated on the clean image")
    reference_axis = None
    for name in MAP_NAMES:
        score = payload["maps"][name]
        axis = np.asarray(score["fractions"], dtype=np.float64)
        if (
            axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
            or axis[0] != 0 or axis[-1] != 1 or not (np.diff(axis) > 0).all()
        ):
            raise ValueError(f"{path}: invalid {name} fraction axis")
        if reference_axis is not None and not np.array_equal(reference_axis, axis):
            raise ValueError(f"{path}: map fraction axes differ")
        reference_axis = axis
        for mode in ("ins", "del"):
            curve = np.asarray(score[f"{mode}_curve"], dtype=np.float64)
            if (
                curve.shape != axis.shape or not np.isfinite(curve).all()
                or (curve < 0).any() or (curve > 1).any()
            ):
                raise ValueError(f"{path}: invalid {name} {mode} probabilities")
            area = float(np.sum((curve[:-1] + curve[1:]) * np.diff(axis) / 2))
            if not math.isclose(area, score[f"{mode}_auc"], rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"{path}: {name} {mode} AUC does not match its curve")
        if not math.isclose(
            score["ins_auc"] - score["del_auc"], score["imd"],
            rel_tol=1e-7, abs_tol=1e-9,
        ):
            raise ValueError(f"{path}: {name} IMD is not Ins AUC - Del AUC")


def configuration(payload):
    context = payload["context"]
    return {
        "strategy": context["strategy"]
    } | {
        "protocol": payload["protocol"],
        "perturbation_metadata": {
            key: value for key, value in (payload.get("perturbation_metadata") or {}).items()
            if key != "gamma"
        },
    }


def approach_name(context):
    if context["algorithm"] == "nsgaii":
        return "nsgaii"
    if context["algorithm"] == "weighted_sum_ga":
        for weight in (0.0, 0.5, 1.0):
            if (
                math.isclose(context["w_m"], weight, abs_tol=1e-9)
                and math.isclose(context["w_s"], 1 - weight, abs_tol=1e-9)
            ):
                return f"ga_{weight:g}"
    raise ValueError(f"Unsupported comparison approach: {context}")


def block_key(context):
    return tuple(context[key] for key in ("model", "explain_method", "eps", "loss_type"))


def sample_key(payload):
    # Class/sample directory names are shared across approach and seed roots.
    return (payload["context"]["seed"], *Path(payload["sample_dir"]).parts[-2:])


def comparison_samples(records):
    buckets = defaultdict(dict)
    clean_samples = {}
    for payload in records:
        context = payload["context"]
        block = block_key(context)
        identity = sample_key(payload)
        approach = approach_name(context)
        key = (*block, approach)
        if identity in buckets[key]:
            raise ValueError(f"Duplicate {approach} evaluation for {block}, {identity}")
        buckets[key][identity] = (payload, "perturbation")
        clean_key = (*block, identity)
        previous = clean_samples.get(clean_key)
        if previous is not None:
            if previous["target_class"] != payload["target_class"]:
                raise ValueError(f"Inconsistent clean target for {clean_key}")
            old_hash = previous.get("clean_image_sha256")
            new_hash = payload.get("clean_image_sha256")
            if old_hash and new_hash and old_hash != new_hash:
                raise ValueError(f"Different clean images for {clean_key}")
            if "localization_provenance" in previous or "localization_provenance" in payload:
                if previous.get("localization_provenance") != payload.get("localization_provenance"):
                    raise ValueError(f"Inconsistent localization Clean/GT for {clean_key}")
                if previous["maps"]["clean"].get("iou") != payload["maps"]["clean"].get("iou"):
                    raise ValueError(f"Inconsistent clean IoU for {clean_key}")
            for field in CURVE_FIELDS:
                left = np.asarray(previous["maps"]["clean"][field])
                right = np.asarray(payload["maps"]["clean"][field])
                if left.shape != right.shape or not np.allclose(left, right, rtol=1e-5, atol=1e-5):
                    raise ValueError(f"Inconsistent clean {field} for {clean_key}")
        else:
            clean_samples[clean_key] = payload
            buckets[(*block, "clean")][identity] = (payload, "clean")
    return buckets


def summarize(records):
    buckets = comparison_samples(records)
    rows = []
    for (model, method, eps, loss, name), entries in sorted(buckets.items()):
        samples = list(entries.values())
        row = {
            "family": model_family(model), "model": model, "explain_method": method,
            "eps": eps, "loss_type": loss, "approach": name,
            "n": len(samples),
            "n_seeds": len({sample["context"]["seed"] for sample, _ in samples}),
        }
        for metric in METRICS:
            values = [sample["maps"][source][metric] for sample, source in samples]
            row[metric] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        axes = [sample["maps"][source]["fractions"] for sample, source in samples]
        if any(not np.array_equal(axes[0], axis) for axis in axes[1:]):
            raise ValueError(f"Different fraction axes in {model}/{method}/{eps}/{loss}/{name}")
        row["fractions"] = axes[0]
        for field in ("ins_curve", "del_curve"):
            row[field] = np.mean(
                [sample["maps"][source][field] for sample, source in samples], axis=0
            ).tolist()
        row["sample_ids"] = [list(identity) for identity in sorted(entries)]
        rows.append(row)
    return rows


def save_group(output_dir, rows, method, family):
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = [
        {key: value for key, value in row.items() if key not in (*CURVE_FIELDS, "sample_ids")}
        for row in rows
    ]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    models = {}
    for row in rows:
        maps = (
            models.setdefault(row["model"], {})
            .setdefault(str(row["eps"]), {})
            .setdefault(row["loss_type"], {})
        )
        maps[row["approach"]] = {
            key: row[key] for key in ("n", "n_seeds", *METRICS,
                                     *(f"{metric}_std" for metric in METRICS))
        }
    write_json(output_dir / "summary.json", {
        "family": family, "explain_method": method,
        "averaging": "sample_weighted_mean", "std": "sample_standard_deviation",
        "models": models,
    })
    stale_table = output_dir / "table.tex"
    if stale_table.exists():
        stale_table.unlink()


def save_curves(output_dir, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {row["approach"]: row for row in rows}
    missing = [name for name in APPROACHES if name not in lookup]
    cohorts = [row["sample_ids"] for row in rows]
    matched = all(ids == cohorts[0] for ids in cohorts[1:])
    complete = not missing and matched
    first = rows[0]
    directory = output_dir / first["model"] / f"eps_{first['eps']}" / first["loss_type"]
    write_json(directory / "curves.json", {
        "block": {key: first[key] for key in ("model", "explain_method", "eps", "loss_type")},
        "complete": complete, "missing_approaches": missing, "matched_samples": matched,
        "averaging": "sample_weighted_mean",
        "approaches": lookup,
    })
    if not complete:
        print(f"[PARTIAL] {directory}: missing={missing}, matched_samples={matched}")
    colors = ("black", "#0072B2", "#E69F00", "#009E73", "#CC79A7")
    for mode, title in (("ins", "Insertion"), ("del", "Deletion")):
        fig, axis = plt.subplots(figsize=(6, 4))
        try:
            for name, color in zip(APPROACHES, colors):
                row = lookup.get(name)
                if row is None:
                    continue
                axis.plot(row["fractions"], row[f"{mode}_curve"], color=color,
                          label=f"{LABELS[name]} (AUC={row[f'{mode}_auc']:.4f}, n={row['n']})")
            axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Fraction of pixels",
                     ylabel="True-label probability",
                     title=f"{title}: {first['model']} / {first['explain_method']}\n"
                           f"eps={first['eps']} / {first['loss_type']}"
                           + ("" if complete else " [PARTIAL]"))
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
            fig.tight_layout()
            for extension in ("png", "pdf"):
                fig.savefig(directory / f"{mode}_curve.{extension}", dpi=200)
        finally:
            plt.close(fig)


def aggregate_records(records, output_dir, plot_blocks=None):
    groups = defaultdict(list)
    seen_samples = set()
    for payload in records:
        validate_record(payload, payload["sample_dir"])
        identity = payload["sample_dir"]
        if identity in seen_samples:
            raise ValueError(f"Duplicate sample evaluation: {identity}")
        seen_samples.add(identity)
        key = json.dumps(configuration(payload), sort_keys=True, allow_nan=False)
        groups[key].append(payload)
    manifest = []
    for key, records in sorted(groups.items()):
        folder = output_dir / ("comparison_" + hashlib.sha256(key.encode()).hexdigest()[:12])
        config = json.loads(key)
        write_json(folder / "configuration.json", config)
        rows = summarize(records)
        by_method = defaultdict(list)
        for row in rows:
            by_method[(row["family"], row["explain_method"])].append(row)
        for (family, method), method_rows in sorted(by_method.items()):
            if not method or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in method):
                raise ValueError(f"Invalid explanation method path: {method!r}")
            directory = folder / family / method
            save_group(directory, method_rows, method, family)
            blocks = defaultdict(list)
            for row in method_rows:
                blocks[block_key(row)].append(row)
            for block, block_rows in sorted(blocks.items()):
                if plot_blocks is None or block in plot_blocks:
                    save_curves(directory, block_rows)
        manifest.append({
            "directory": folder.name, "configuration": config,
            "num_samples": len(records), "rows": rows,
        })
    write_json(output_dir / "summary.json", {
        "num_samples": len(seen_samples), "num_configurations": len(groups),
        "averaging": "sample_weighted_mean",
        "clean_deduplication": "seed/model/explainer/eps/loss/class/sample",
        "configurations": manifest,
    })
    print(f"Aggregated {len(seen_samples)} evaluations into {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("faithfulness_aggregates"))
    args = parser.parse_args()
    paths = set()
    for root in args.input_dirs:
        if not root.is_dir():
            raise NotADirectoryError(root)
        paths.update(path.resolve() for path in root.rglob("faithfulness.json"))
    if not paths:
        raise ValueError("No faithfulness.json evaluations found")
    records = []
    for path in sorted(paths):
        with path.open(encoding="utf-8") as handle:
            records.append(json.load(handle))
    aggregate_records(records, args.output_dir)


if __name__ == "__main__":
    main()
