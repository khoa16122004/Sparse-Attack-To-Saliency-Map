"""Aggregate faithfulness.json files and render Clean/Adversarial/Perturbation tables."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__:
    from .run_score import _format_explain_latex_name, _format_model_latex_name, _sort_model_keys
else:
    from run_score import _format_explain_latex_name, _format_model_latex_name, _sort_model_keys


MAP_NAMES = ("clean", "adversarial", "perturbation")
METRICS = ("ins_auc", "del_auc", "imd")
LOSSES = ("margin_loss", "negative_cross_entropy_saliency")
LABELS = {"clean": "Clean", "adversarial": "Adversarial", "perturbation": "Perturbation"}


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
        key: context[key] for key in ("algorithm", "strategy", "w_m", "w_s")
    } | {
        "protocol": payload["protocol"],
        "perturbation_metadata": payload.get("perturbation_metadata"),
    }


def summarize(records):
    buckets = defaultdict(list)
    for payload in records:
        context = payload["context"]
        for name in MAP_NAMES:
            key = (
                context["model"], context["explain_method"], context["eps"],
                context["loss_type"], name,
            )
            buckets[key].append(payload)
    rows = []
    for (model, method, eps, loss, name), samples in sorted(buckets.items()):
        row = {
            "family": model_family(model), "model": model, "explain_method": method,
            "eps": eps, "loss_type": loss, "map": name,
            "n": len(samples),
            "n_seeds": len({sample["context"]["seed"] for sample in samples}),
        }
        for metric in METRICS:
            values = [sample["maps"][name][metric] for sample in samples]
            row[metric] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def latex_table(rows, budgets, method, family):
    unknown = {row["loss_type"] for row in rows} - set(LOSSES)
    if unknown:
        raise ValueError(f"Unsupported table loss functions: {sorted(unknown)}")
    columns = 6 * len(budgets)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Faithfulness of Clean, Adversarial, and Perturbation maps "
            f"({_format_explain_latex_name(method)}, {family}). "
            r"All maps are evaluated on the clean image for the original class.}"
        ),
        r"\label{tab:faithfulness_" + family.lower() + "_" + method + "}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{cc|" + "|".join("cccccc" for _ in budgets) + "}",
        r"\toprule",
        r"\multirow{3}{*}{Model} & \multirow{3}{*}{Saliency Map} & "
        + " & ".join(r"\multicolumn{6}{c}{$\epsilon=" + str(eps) + "$}" for eps in budgets)
        + r" \\",
        " ".join(
            rf"\cmidrule(lr){{{3 + 6 * i}-{8 + 6 * i}}}" for i in range(len(budgets))
        ),
        "& & " + " & ".join(
            [r"\multicolumn{3}{c}{Margin Loss}", r"\multicolumn{3}{c}{Log-Likelihood}"]
            * len(budgets)
        ) + r" \\",
        " ".join(
            rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(2 * len(budgets))
        ),
        "& & " + " & ".join(
            [r"Ins.$\uparrow$", r"Del.$\downarrow$", "IMD"] * (columns // 3)
        ) + r" \\",
        r"\midrule",
    ]
    lookup = {
        (row["model"], row["eps"], row["loss_type"], row["map"]): row for row in rows
    }
    models = _sort_model_keys(sorted({row["model"] for row in rows}))
    for index, model in enumerate(models):
        for map_index, name in enumerate(MAP_NAMES):
            model_cell = (
                r"\multirow{3}{*}{" + _format_model_latex_name(model) + "}"
                if map_index == 0 else ""
            )
            cells = []
            for eps in budgets:
                for loss in LOSSES:
                    row = lookup.get((model, eps, loss, name))
                    cells.extend(
                        [f"{row[metric]:.4f}" for metric in METRICS]
                        if row is not None else ["--"] * 3
                    )
            lines.append(model_cell + " & " + LABELS[name] + " & " + " & ".join(cells) + r" \\")
        if index != len(models) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def save_group(output_dir, rows, budgets, method, family):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    models = {}
    for row in rows:
        maps = (
            models.setdefault(row["model"], {})
            .setdefault(str(row["eps"]), {})
            .setdefault(row["loss_type"], {})
        )
        maps[row["map"]] = {
            key: row[key] for key in ("n", "n_seeds", *METRICS,
                                     *(f"{metric}_std" for metric in METRICS))
        }
    write_json(output_dir / "summary.json", {
        "family": family, "explain_method": method,
        "averaging": "sample_weighted_mean", "std": "sample_standard_deviation",
        "models": models,
    })
    (output_dir / "table.tex").write_text(
        latex_table(rows, budgets, method, family), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("faithfulness_aggregates"))
    parser.add_argument("--eps", type=int, nargs="+", default=[20, 50, 100],
                        help="Budget columns in LaTeX (JSON/CSV retain all budgets)")
    args = parser.parse_args()
    if any(eps <= 0 for eps in args.eps) or len(set(args.eps)) != len(args.eps):
        parser.error("--eps must contain distinct positive budgets")
    paths = set()
    for root in args.input_dirs:
        if not root.is_dir():
            raise NotADirectoryError(root)
        paths.update(path.resolve() for path in root.rglob("faithfulness.json"))
    if not paths:
        raise ValueError("No faithfulness.json evaluations found")
    groups = defaultdict(list)
    seen_samples = set()
    for path in sorted(paths):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        validate_record(payload, path)
        identity = payload["sample_dir"]
        if identity in seen_samples:
            raise ValueError(f"Duplicate sample evaluation: {identity}")
        seen_samples.add(identity)
        key = json.dumps(configuration(payload), sort_keys=True, allow_nan=False)
        groups[key].append(payload)
    manifest = []
    for index, (key, records) in enumerate(sorted(groups.items()), start=1):
        folder = args.output_dir / f"config_{index:03d}"
        config = json.loads(key)
        write_json(folder / "configuration.json", config)
        rows = summarize(records)
        by_method = defaultdict(list)
        for row in rows:
            by_method[(row["family"], row["explain_method"])].append(row)
        for (family, method), method_rows in sorted(by_method.items()):
            if not method or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in method):
                raise ValueError(f"Invalid explanation method path: {method!r}")
            save_group(folder / family / method, method_rows, args.eps, method, family)
        manifest.append({
            "directory": folder.name, "configuration": config,
            "num_samples": len(records), "rows": rows,
        })
    write_json(args.output_dir / "summary.json", {
        "num_samples": len(paths), "num_configurations": len(groups),
        "averaging": "sample_weighted_mean",
        "table_budgets": args.eps,
        "configurations": manifest,
    })
    print(f"Aggregated {len(paths)} samples into {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
