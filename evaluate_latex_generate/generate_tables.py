"""Generate the paper tables from attack runs and per-sample faithfulness scores."""

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_script import run_aggregate_faithfulness as faithfulness
from evaluate_script import run_score


TEMPLATE_DIR = Path(__file__).resolve().parent
APPROACHES = ("ga_0", "ga_0.5", "ga_1", "nsgaii")
METRICS = {
    "attack": ("asr", "sro", "h_score"),
    "faithfulness": ("ins_auc", "del_auc", "imd"),
}
MODELS = {
    "attack": (
        ("resnet18", "vgg16", "densenet121"),
        ("vit_b_32", "vit_b_16", "swin_v2"),
    ),
    "faithfulness": (
        ("resnet18", "vgg16", "densenet121"),
        ("vit_b_32", "vit_b_16", "swin_v2"),
    ),
}
METHODS = (
    ("simple_gradient", "input_gradient", "grad_cam", "xgrad_cam"),
    ("raw_attention", "attention_rollout", "attention_flow", "partial_lrp"),
)
METHOD_ALIASES = {
    "xgradcam": "xgrad_cam", "x_grad_cam": "xgrad_cam",
    "rollout": "attention_rollout", "flow": "attention_flow",
    "lrp": "partial_lrp",
}


def canonical_model(name):
    name = name.strip().lower().replace("-", "_")
    return {
        "resnet_50": "resnet50", "vgg_16": "vgg16",
        "densenet_121": "densenet121", "swin_v2_t": "swin_v2",
    }.get(name, name)


def cell_key(model, method, approach):
    return (canonical_model(model), METHOD_ALIASES.get(method, method), approach)


def config_key(config):
    return json.dumps(config, sort_keys=True, allow_nan=False)


def add_seed(cells, key, seed, values, sample_ids=None):
    seed = str(seed)
    if seed in cells[key]:
        raise ValueError(f"Duplicate run for {key}, seed={seed}; do not mix repeated runs")
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError(f"Non-finite metric for {key}, seed={seed}: {values}")
    cells[key][seed] = {"metrics": values, "sample_ids": sample_ids}


def attack_groups(runs):
    groups = {}
    seen_paths = set()
    for run in runs:
        path = run.run_dir.resolve()
        if path in seen_paths:
            raise ValueError(f"Duplicate attack run: {path}")
        seen_paths.add(path)
        if run.seed in ("", "default", "unknown", "None"):
            raise ValueError(f"Explicit seed required for attack run: {run.run_dir}")
        if run.num_samples_ok <= 0:
            raise ValueError(f"No successful sample records in {run.run_dir}")
        config = {
            "strategy": run.strategy, "eps": run.eps, "loss_type": run.loss_type,
        }
        cells = groups.setdefault(config_key(config), defaultdict(dict))
        approach = faithfulness.approach_name({
            "algorithm": run_score._normalize_algorithm(run.algorithm),
            "w_m": run.w_m, "w_s": run.w_s,
        })
        # H-score must be computed within each seed, before averaging seeds.
        values = {
            "asr": run.asr, "sro": run.spearman,
            "h_score": run_score._h_score(run.asr, run.spearman),
        }
        add_seed(cells, cell_key(run.model, run.explain_method, approach), run.seed, values)
    return groups


def attach_localization(records, localization_records):
    indexed = {}
    for record in localization_records:
        key = Path(record["sample_dir"]).resolve()
        if key in indexed:
            raise ValueError(f"Duplicate localization sample: {key}")
        indexed[key] = record
    result = []
    for original in records:
        path = Path(original["sample_dir"]).resolve()
        if path not in indexed:
            raise ValueError(f"No localization IoU for faithfulness sample: {path}")
        localization = indexed.pop(path)
        if original["context"] != localization["context"]:
            raise ValueError(f"Faithfulness/localization context mismatch: {path}")
        record = copy.deepcopy(original)
        for source in ("clean", "perturbation"):
            value = localization[f"{source}_iou"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {source} IoU for {path}: {value}")
            record["maps"][source]["iou"] = value
        record["localization_protocol"] = localization["protocol"]
        record["localization_provenance"] = {
            key: localization[key]
            for key in ("clean_rgb_sha256", "gt_mask_sha256", "clean_binary_sha256")
        }
        for key in ("clean_baseline_dir", "clean_baseline_seed"):
            if key in localization:
                record["localization_provenance"][key] = localization[key]
        result.append(record)
    if indexed:
        raise ValueError(f"Localization samples have no faithfulness scores: {list(indexed)[:3]}")
    return result


def faithfulness_groups(records, *, include_iou=False):
    grouped = defaultdict(list)
    for record in records:
        faithfulness.validate_record(record, record["sample_dir"])
        context = record["context"]
        if str(context["seed"]) in ("", "default", "unknown", "None"):
            raise ValueError(f"Explicit seed required: {record['sample_dir']}")
        config = faithfulness.configuration(record) | {
            "eps": context["eps"], "loss_type": context["loss_type"],
        }
        if include_iou:
            config["localization_protocol"] = record["localization_protocol"]
        grouped[config_key(config)].append(record)
    groups = {}
    for config, payloads in grouped.items():
        cells = defaultdict(dict)
        for (model, method, _eps, _loss, approach), samples in (
            faithfulness.comparison_samples(payloads).items()
        ):
            by_seed = defaultdict(list)
            for identity, (payload, source) in samples.items():
                by_seed[str(identity[0])].append((identity[1:], payload["maps"][source]))
            for seed, entries in by_seed.items():
                values = {
                    metric: mean(score[metric] for _, score in entries)
                    for metric in (*METRICS["faithfulness"], *(("iou",) if include_iou else ()))
                }
                add_seed(
                    cells, cell_key(model, method, approach), seed, values,
                    sorted([list(identity) for identity, _ in entries]),
                )
        groups[config] = cells
    return groups


def summarize_cells(cells, metrics):
    summary = {}
    for key, seeds in cells.items():
        stats = {}
        for metric in metrics:
            values = [entry["metrics"][metric] for entry in seeds.values()]
            stats[metric] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else None,
                "n_seeds": len(values),
                "seed_values": {
                    seed: entry["metrics"][metric] for seed, entry in sorted(seeds.items())
                },
            }
        summary[key] = stats
    return summary


def paired_comparison(ours, baseline, metric, higher, ttest):
    seeds = sorted(set(ours) & set(baseline))
    result = {"seeds": seeds, "n_pairs": len(seeds), "p_value": None, "improved": False}
    if len(seeds) < 2:
        return result | {"status": "insufficient_pairs"}
    for seed in seeds:
        if ours[seed]["sample_ids"] != baseline[seed]["sample_ids"]:
            return result | {"status": "unmatched_samples"}
    differences = [
        (ours[seed]["metrics"][metric] - baseline[seed]["metrics"][metric])
        * (1 if higher else -1)
        for seed in seeds
    ]
    result["improved"] = mean(differences) > 0
    # Avoid scipy's precision-loss warning for exactly constant paired differences.
    if all(value == differences[0] for value in differences):
        p_value = 0.0 if differences[0] != 0 else 1.0
    else:
        p_value = float(ttest(differences, [0.0] * len(seeds)).pvalue)
        if not math.isfinite(p_value):
            raise ValueError(f"Non-finite paired t-test result for {metric}, seeds={seeds}")
    return result | {"p_value": p_value, "status": "ok"}


def render_table(kind, cells, *, use_ttest=True, alpha=0.05, correction="none", precision=4,
                 include_iou=False, sd_format="pm"):
    if sd_format not in ("pm", "subscript"):
        raise ValueError(f"Unsupported SD format: {sd_format}")
    if include_iou and kind != "faithfulness":
        raise ValueError("IoU columns are only supported for faithfulness tables")
    if use_ttest:
        try:
            from scipy.stats import ttest_rel
        except ImportError as exc:
            raise RuntimeError("T-tests require scipy; install scipy or pass --no-ttest") from exc
    else:
        ttest_rel = None
    metrics = (*METRICS[kind], *(("iou",) if include_iou else ()))
    summary = summarize_cells(cells, metrics)
    approaches = (("clean",) + APPROACHES) if kind == "faithfulness" else APPROACHES
    tests = []
    significant = set()
    for family, models in enumerate(MODELS[kind]):
        for method in METHODS[family]:
            for model in models:
                ours = cells.get((model, method, "nsgaii"), {})
                for metric in metrics:
                    if not use_ttest:
                        continue
                    comparisons = []
                    for baseline in APPROACHES[:-1]:
                        result = paired_comparison(
                            ours, cells.get((model, method, baseline), {}),
                            metric, metric not in ("sro", "del_auc"), ttest_rel,
                        )
                        tests.append({
                            "model": model, "explain_method": method, "metric": metric,
                            "baseline": baseline, **result,
                        })
                        comparisons.append(result)
                    threshold = alpha / 3 if correction == "bonferroni" else alpha
                    if all(
                        result["status"] == "ok" and result["improved"]
                        and result["p_value"] < threshold
                        for result in comparisons
                    ):
                        significant.add((model, method, metric))

    template = TEMPLATE_DIR / ("perform_attack.text" if kind == "attack" else "faithfulness.text")
    lines = template.read_text(encoding="utf-8").splitlines()
    seed_counts = sorted({len(seeds) for key, seeds in cells.items() if key in {
        (model, method, approach)
        for family, models in enumerate(MODELS[kind])
        for model in models for method in METHODS[family] for approach in approaches
    }})
    counts = ", ".join(map(str, seed_counts)) or "0"
    caption = (
        r"Each seed is averaged over its samples first; results are the "
        r"equally weighted mean $\pm$ sample standard deviation across seeds "
        f"(available seeds per cell: {counts}). "
        r"Missing values and undefined standard deviations are shown as --. "
        r"The best mean for each explainer, model, and metric is in \textbf{bold}."
    )
    if sd_format == "subscript":
        caption = caption.replace(
            r"mean $\pm$ sample standard deviation",
            r"mean with sample standard deviation as a lower-right subscript",
        )
    if use_ttest:
        threshold = alpha / 3 if correction == "bonferroni" else alpha
        caption += (
            r" \textcolor{blue}{Blue} marks Ours only when it improves over all "
            r"three GASS baselines in two-sided paired t-tests on matching seeds "
            f"($p < {threshold:.6g}$"
            + (", Bonferroni corrected" if correction == "bonferroni" else ", uncorrected")
            + r"); at least two matched seed pairs are required."
        )
    family = -1
    method_index = -1
    row_index = 0
    rendered = []
    for line in lines:
        if include_iou:
            line = line.replace("{ll|ccc|ccc|ccc}", "{ll|cccc|cccc|cccc}")
            line = line.replace(r"\multicolumn{3}", r"\multicolumn{4}")
            line = line.replace("{3-5}", "{3-6}").replace("{6-8}", "{7-10}").replace("{9-11}", "{11-14}")
            line = line.replace(r"IMD $\uparrow$", r"IMD $\uparrow$ & IoU $\uparrow$")
        if line.startswith(r"\caption{"):
            prefix = line.split("All experiments", 1)[0]
            if include_iou:
                prefix = prefix.replace(
                    r", and IMD $\uparrow$ & IoU $\uparrow$.",
                    r", IMD $\uparrow$, and localization IoU $\uparrow$ against SAM3 pseudo-GT.",
                )
            rendered.append(prefix + caption + "}")
            continue
        if line.startswith(r"\textit{"):
            family += 1
            method_index = -1
        if re.match(r"\\multirow\{[45]\}", line):
            method_index += 1
            row_index = 0
        if line.startswith("& ") and method_index >= 0:
            method = METHODS[family][method_index]
            approach = approaches[row_index]
            row_index += 1
            values = []
            for model in MODELS[kind][family]:
                stats = summary.get((model, method, approach))
                for metric in metrics:
                    if stats is None:
                        values.append("--")
                        continue
                    stat = stats[metric]
                    candidates = [
                        summary[(model, method, name)][metric]["mean"]
                        for name in approaches if (model, method, name) in summary
                    ]
                    best = (min if metric in ("sro", "del_auc") else max)(candidates)
                    sd = "--" if stat["std"] is None else f"{stat['std']:.{precision}f}"
                    value = f"{stat['mean']:.{precision}f}"
                    if sd_format == "subscript":
                        value += "_{" + sd + "}"
                    else:
                        value += r" \pm " + sd
                    if stat["mean"] == best:
                        value = r"\mathbf{" + value + "}"
                    if approach == "nsgaii" and (model, method, metric) in significant:
                        value = r"\textcolor{blue}{" + value + "}"
                    values.append("$" + value + "$")
            label = line.split("&")[1].strip()
            line = "& " + label + " & " + " & ".join(values) + r" \\"
        rendered.append(line)
    used = {
        (model, method, approach)
        for family, models in enumerate(MODELS[kind])
        for model in models for method in METHODS[family] for approach in approaches
    }
    report = {
        "averaging": "equal_weight_seed_means", "std": "sample_std_ddof_1",
        "ttest": {
            "enabled": use_ttest, "alpha": alpha, "correction": correction,
            "alternative": "two-sided", "comparisons": tests,
        },
        "cells": [
            {"model": key[0], "explain_method": key[1], "approach": key[2], "metrics": stats}
            for key, stats in sorted(summary.items()) if key in used
        ],
        "missing_cells": [list(key) for key in sorted(used - cells.keys())],
        "excluded_cells": [list(key) for key in sorted(cells.keys() - used)],
    }
    return "\n".join(rendered) + "\n", report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--attack-input-dirs", nargs="+", type=Path,
                        help="Raw attack roots, read using run_score")
    source.add_argument("--all-runs-json", nargs="+", type=Path,
                        help="Cached all_runs.json files produced by run_score")
    source.add_argument("--faithfulness-input-dirs", nargs="+", type=Path,
                        help="Roots containing per-sample faithfulness.json")
    parser.add_argument("--localization-records", type=Path,
                        help="Exact records JSON from run_localization_iou; adds IoU to faithfulness tables")
    parser.add_argument("--output-dir", type=Path, default=TEMPLATE_DIR / "generated")
    parser.add_argument("--use-ttest", dest="use_ttest", action="store_true", default=True)
    parser.add_argument("--no-ttest", dest="use_ttest", action="store_false")
    parser.add_argument("--ttest-alpha", type=float, default=0.05)
    parser.add_argument("--ttest-correction", choices=("none", "bonferroni"), default="none")
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--sd-format", choices=("pm", "subscript"), default="pm",
                        help="Display SD after plus/minus or as a lower-right subscript")
    parser.add_argument("--eps", nargs="+", type=int)
    parser.add_argument("--loss-type")
    parser.add_argument("--strategy")
    args = parser.parse_args(argv)
    if args.localization_records and not args.faithfulness_input_dirs:
        parser.error("--localization-records requires --faithfulness-input-dirs")
    if not 0 < args.ttest_alpha < 1:
        parser.error("--ttest-alpha must be between 0 and 1")
    if not 0 <= args.precision <= 12:
        parser.error("--precision must be between 0 and 12")
    if args.faithfulness_input_dirs:
        paths = set()
        for root in args.faithfulness_input_dirs:
            if not root.is_dir():
                raise NotADirectoryError(root)
            paths.update(path.resolve() for path in root.rglob("faithfulness.json"))
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths)]
        if args.localization_records:
            payload = json.loads(args.localization_records.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 1:
                raise ValueError(f"{args.localization_records}: unsupported localization record schema")
            # Apply the same filters used by the CPU localization job before joining.
            records = [
                record for record in records
                if (args.eps is None or record["context"]["eps"] in args.eps)
                and (args.loss_type is None or record["context"]["loss_type"] == args.loss_type)
                and (args.strategy is None or record["context"]["strategy"] == args.strategy)
            ]
            records = attach_localization(records, payload["records"])
        groups = faithfulness_groups(records, include_iou=bool(args.localization_records))
        kind = "faithfulness"
    else:
        runs = []
        if args.all_runs_json:
            for path in dict.fromkeys(path.resolve() for path in args.all_runs_json):
                runs.extend(run_score._load_runs_from_json(path))
        else:
            for root in dict.fromkeys(path.resolve() for path in args.attack_input_dirs):
                runs.extend(run_score._load_all_runs(root))
        groups = attack_groups(runs)
        kind = "attack"
    groups = {
        key: cells for key, cells in groups.items()
        if (args.eps is None or json.loads(key)["eps"] in args.eps)
        and (args.loss_type is None or json.loads(key)["loss_type"] == args.loss_type)
        and (args.strategy is None or json.loads(key)["strategy"] == args.strategy)
    }
    if not groups:
        raise ValueError("No matching evaluations found")
    for key, cells in sorted(groups.items()):
        latex, report = render_table(
            kind, cells, use_ttest=args.use_ttest, alpha=args.ttest_alpha,
            correction=args.ttest_correction, precision=args.precision,
            include_iou=bool(args.localization_records),
            sd_format=args.sd_format,
        )
        if args.localization_records:
            protocol = json.loads(key)["localization_protocol"]
            latex = latex.replace(
                "Each seed is averaged",
                f"Localization uses top {protocol['top_k_percent']:g}"
                + r"\% pixels against SAM3 pseudo-GT. Each seed is averaged",
            )
        if not report["cells"]:
            raise ValueError(f"No data matches the fixed {kind} template models/explainers: {key}")
        report["configuration"] = json.loads(key)
        folder = args.output_dir / (kind + "_" + hashlib.sha256(key.encode()).hexdigest()[:12])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "table.tex").write_text(latex, encoding="utf-8")
        (folder / "statistics.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        print(f"[OK] {folder}: {len(report['cells'])} cells, "
              f"{len(report['missing_cells'])} missing, "
              f"{len(report['excluded_cells'])} excluded (see statistics.json)")


if __name__ == "__main__":
    main()
