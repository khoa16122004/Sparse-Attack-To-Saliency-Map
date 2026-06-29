#!/usr/bin/env python3
import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import wandb


SUCCESS_MARGIN_THRESHOLD = 0.0


@dataclass
class ApproachCurves:
    label: str
    model: str
    approach: str
    eps: str
    strategy: str
    fitness: str
    algorithm: str
    combo_key: str
    source_root: str
    num_ok: int
    num_total: int
    margin_mean: List[float]
    saliency_mean: List[float]
    asr_cumulative: List[float]
    success_count: List[int]
    num_curve_samples: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build W&B dashboard for iteration-wise comparison across approaches "
            "from multiple output roots"
        )
    )
    parser.add_argument(
        "--input-roots",
        type=Path,
        nargs="+",
        required=True,
        help="One or more roots like compare_loss_50 saliency_guided_outputs_50",
    )
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity/team")
    parser.add_argument("--group", type=str, default="iteration_dashboard", help="W&B group name")
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=["dashboard", "iteration", "sparse_attack"],
        help="W&B tags",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously refresh dashboard by rescanning input roots",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=120,
        help="Refresh interval in seconds when --watch is enabled",
    )
    return parser.parse_args()


def parse_approach_tag(tag: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for part in tag.split("__"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        parsed[key] = value
    return parsed


def normalize_fitness(value: Optional[str]) -> str:
    if not value:
        return "margin_saliency"
    value = str(value).strip().lower()
    if value in {"margin", "margin_loss", "margin_saliency"}:
        return "margin_saliency"
    if value in {"cross_entropy_saliency", "negative_cross_entropy_saliency"}:
        return "cross_entropy_saliency"
    return value


def normalize_algorithm(value: Optional[str]) -> str:
    if not value:
        return "weighted_sum_ga"
    value = str(value).strip().lower()
    if value in {"weighted_sum_ga", "weighted_sum", "ga", "default"}:
        return "weighted_sum_ga"
    if value in {"nsgaii", "nsga2", "nsga_ii"}:
        return "nsgaii"
    return value


def fitness_display_name(value: str) -> str:
    if value == "margin_saliency":
        return "marginloss"
    if value == "cross_entropy_saliency":
        return "crossentropy"
    return value


def _eps_sort_key(eps: str):
    try:
        return (0, float(eps))
    except (TypeError, ValueError):
        return (1, str(eps))


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_output_dir(item: dict, approach_dir: Path) -> Optional[Path]:
    output_dir = item.get("output_dir")
    if not output_dir:
        return None

    candidate = Path(output_dir)
    if candidate.is_absolute():
        return candidate

    root_dir = approach_dir.parent.parent
    return root_dir / candidate


def _to_float_list(values) -> Optional[List[float]]:
    if not isinstance(values, list):
        return None
    out = []
    for value in values:
        if not isinstance(value, (int, float)):
            return None
        out.append(float(value))
    return out


def _load_history_from_txt(txt_path: Path) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    if not txt_path.exists():
        return None, None

    margin = []
    saliency = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                margin.append(float(parts[0]))
                saliency.append(float(parts[1]))
            except ValueError:
                continue

    if not margin or len(margin) != len(saliency):
        return None, None

    return margin, saliency


def _load_item_history(item: dict, approach_dir: Path) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    margin = _to_float_list(item.get("history_margin"))
    saliency = _to_float_list(item.get("history_saliency"))
    if margin is not None and saliency is not None and len(margin) == len(saliency):
        return margin, saliency

    output_dir = _resolve_output_dir(item, approach_dir)
    if output_dir is None:
        return None, None

    return _load_history_from_txt(output_dir / "history_scores.txt")


def _first_success_iteration(item: dict, margin_curve: List[float]) -> Optional[int]:
    raw = item.get("first_success_iteration")
    if isinstance(raw, int) and raw >= 0:
        return raw

    for idx, margin in enumerate(margin_curve):
        if margin <= SUCCESS_MARGIN_THRESHOLD:
            return idx
    return None


def _mean_curve(curves: List[List[float]]) -> List[float]:
    if not curves:
        return []

    max_len = max(len(curve) for curve in curves)
    out = []
    for idx in range(max_len):
        values = [curve[idx] for curve in curves if idx < len(curve)]
        out.append(float(sum(values) / len(values)))
    return out


def _asr_curve(first_success_iters: List[Optional[int]], n_iters: int) -> Tuple[List[float], List[int]]:
    if n_iters <= 0:
        return [], []

    total = len(first_success_iters)
    if total == 0:
        return [0.0] * n_iters, [0] * n_iters

    asr = []
    success_count = []
    for iteration in range(n_iters):
        count = 0
        for first_it in first_success_iters:
            if first_it is not None and first_it <= iteration:
                count += 1
        success_count.append(count)
        asr.append(float(count / total))

    return asr, success_count


def _rebuild_report_from_summaries(approach_dir: Path) -> Optional[dict]:
    summary_files = sorted(approach_dir.glob("*/*/summary.json"))
    if not summary_files:
        return None

    results = []
    for summary_file in summary_files:
        try:
            item = _load_json(summary_file)
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        if "output_dir" not in item:
            item["output_dir"] = str(summary_file.parent)
        if "class" not in item:
            item["class"] = summary_file.parent.parent.name
        results.append(item)

    if not results:
        return None

    return {"results": results, "_generated_from": "summary_files"}


def _load_report_with_fallback(approach_dir: Path) -> Optional[dict]:
    report_path = approach_dir / "batch_report.json"
    if report_path.exists():
        try:
            report = _load_json(report_path)
            if isinstance(report, dict):
                ok_items = [
                    item for item in report.get("results", [])
                    if isinstance(item, dict) and item.get("status") == "ok"
                ]
                if ok_items:
                    return report
        except Exception:
            pass

    return _rebuild_report_from_summaries(approach_dir)


def _collect_from_approach(model_name: str, source_root: Path, approach_dir: Path) -> Optional[ApproachCurves]:
    report = _load_report_with_fallback(approach_dir)
    if report is None:
        return None

    ok_items = [item for item in report.get("results", []) if item.get("status") == "ok"]
    if not ok_items:
        return None

    margin_curves = []
    saliency_curves = []
    first_success_iters = []

    for item in ok_items:
        margin, saliency = _load_item_history(item, approach_dir)
        if margin is None or saliency is None:
            continue
        margin_curves.append(margin)
        saliency_curves.append(saliency)
        first_success_iters.append(_first_success_iteration(item, margin))

    if not margin_curves:
        return None

    margin_mean = _mean_curve(margin_curves)
    saliency_mean = _mean_curve(saliency_curves)
    n_iters = max(len(margin_mean), len(saliency_mean))
    asr_cumulative, success_count = _asr_curve(first_success_iters, n_iters)

    parsed = parse_approach_tag(approach_dir.name)
    eps = parsed.get("eps", "unknown")
    strategy = parsed.get("strategy", "unknown")
    fitness = normalize_fitness(parsed.get("fit"))
    algorithm = normalize_algorithm(parsed.get("algo"))
    combo_key = f"{algorithm}+{strategy}+{fitness_display_name(fitness)}"

    return ApproachCurves(
        label=f"{combo_key}@{source_root.name}",
        model=model_name,
        approach=approach_dir.name,
        eps=eps,
        strategy=strategy,
        fitness=fitness,
        algorithm=algorithm,
        combo_key=combo_key,
        source_root=source_root.name,
        num_ok=len(ok_items),
        num_total=len(report.get("results", [])),
        margin_mean=margin_mean,
        saliency_mean=saliency_mean,
        asr_cumulative=asr_cumulative,
        success_count=success_count,
        num_curve_samples=len(margin_curves),
    )


def collect_all_models(input_roots: List[Path]) -> Dict[str, Dict[str, List[ApproachCurves]]]:
    model_eps_to_curves: Dict[str, Dict[str, List[ApproachCurves]]] = {}

    for root in input_roots:
        for model_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
            model_name = model_dir.name
            for approach_dir in sorted([path for path in model_dir.iterdir() if path.is_dir()]):
                curves = _collect_from_approach(model_name, root, approach_dir)
                if curves is None:
                    continue
                model_eps_to_curves.setdefault(model_name, {})
                model_eps_to_curves[model_name].setdefault(curves.eps, []).append(curves)

    return model_eps_to_curves


def _pad_to(values: List[float], size: int) -> List[float]:
    if len(values) >= size:
        return values[:size]
    return values + [float("nan")] * (size - len(values))


def _merge_curves_by_combo(approaches: List[ApproachCurves]) -> List[ApproachCurves]:
    grouped: Dict[str, List[ApproachCurves]] = {}
    for item in approaches:
        grouped.setdefault(item.combo_key, []).append(item)

    merged: List[ApproachCurves] = []
    for combo_key, items in grouped.items():
        margin_mean = _mean_curve([it.margin_mean for it in items])
        saliency_mean = _mean_curve([it.saliency_mean for it in items])
        max_len = max(len(margin_mean), len(saliency_mean)) if (margin_mean or saliency_mean) else 0

        asr_values = []
        success_values = []
        for idx in range(max_len):
            asr_i = [it.asr_cumulative[idx] for it in items if idx < len(it.asr_cumulative)]
            success_i = [it.success_count[idx] for it in items if idx < len(it.success_count)]
            asr_values.append(float(sum(asr_i) / len(asr_i)) if asr_i else float("nan"))
            success_values.append(int(round(sum(success_i) / len(success_i))) if success_i else 0)

        num_ok = sum(it.num_ok for it in items)
        num_total = sum(it.num_total for it in items)
        num_curve_samples = sum(it.num_curve_samples for it in items)
        source_root = "+".join(sorted(set(it.source_root for it in items)))

        first = items[0]
        merged.append(
            ApproachCurves(
                label=combo_key,
                model=first.model,
                approach=first.approach,
                eps=first.eps,
                strategy=first.strategy,
                fitness=first.fitness,
                algorithm=first.algorithm,
                combo_key=combo_key,
                source_root=source_root,
                num_ok=num_ok,
                num_total=num_total,
                margin_mean=margin_mean,
                saliency_mean=saliency_mean,
                asr_cumulative=asr_values,
                success_count=success_values,
                num_curve_samples=num_curve_samples,
            )
        )

    return sorted(merged, key=lambda x: x.combo_key)


def log_model_dashboard(
    args,
    model_name: str,
    eps: str,
    approaches: List[ApproachCurves],
    scan_index: Optional[int] = None,
):
    approaches = _merge_curves_by_combo(approaches)
    run_id_src = f"{args.project}:{args.group}:{model_name}:eps-{eps}:dashboard".encode("utf-8")
    run_id = hashlib.md5(run_id_src).hexdigest()

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group,
        name=f"dashboard__{model_name}__eps-{eps}",
        id=run_id,
        resume="allow",
        job_type="dashboard",
        config={
            "model": model_name,
            "eps": eps,
            "input_roots": [str(path) for path in args.input_roots],
            "num_approaches": len(approaches),
        },
        tags=args.tags + ["model_dashboard"],
        reinit=True,
        mode=args.mode,
    )

    columns = [
        "approach_label",
        "algorithm",
        "strategy",
        "fitness",
        "eps",
        "source_root",
        "approach",
        "num_ok",
        "num_total",
        "num_curve_samples",
        "final_asr",
        "final_margin_mean",
        "final_saliency_mean",
    ]
    data = []
    for item in approaches:
        data.append([
            item.label,
            item.algorithm,
            item.strategy,
            item.fitness,
            item.eps,
            item.source_root,
            item.approach,
            item.num_ok,
            item.num_total,
            item.num_curve_samples,
            item.asr_cumulative[-1] if item.asr_cumulative else None,
            item.margin_mean[-1] if item.margin_mean else None,
            item.saliency_mean[-1] if item.saliency_mean else None,
        ])

    run.log({"summary/table": wandb.Table(columns=columns, data=data)})

    if scan_index is not None:
        run.summary["dashboard/scan_index"] = scan_index

    max_len = max(
        max((len(a.asr_cumulative) for a in approaches), default=0),
        max((len(a.margin_mean) for a in approaches), default=0),
        max((len(a.saliency_mean) for a in approaches), default=0),
    )

    if max_len > 0:
        xs = list(range(max_len))
        keys = [a.label for a in approaches]

        asr_ys = [_pad_to(a.asr_cumulative, max_len) for a in approaches]
        margin_ys = [_pad_to(a.margin_mean, max_len) for a in approaches]
        saliency_ys = [_pad_to(a.saliency_mean, max_len) for a in approaches]

        run.log(
            {
                "chart/asr_by_iteration": wandb.plot.line_series(
                    xs=xs,
                    ys=asr_ys,
                    keys=keys,
                    title=f"{model_name} (eps={eps}) - ASR by Iteration",
                    xname="Iteration",
                ),
                "chart/margin_mean_by_iteration": wandb.plot.line_series(
                    xs=xs,
                    ys=margin_ys,
                    keys=keys,
                    title=f"{model_name} (eps={eps}) - Mean Margin Loss by Iteration",
                    xname="Iteration",
                ),
                "chart/saliency_mean_by_iteration": wandb.plot.line_series(
                    xs=xs,
                    ys=saliency_ys,
                    keys=keys,
                    title=f"{model_name} (eps={eps}) - Mean Saliency Loss by Iteration",
                    xname="Iteration",
                ),
            }
        )

        for idx in range(max_len):
            payload = {"iteration": idx}
            for approach in approaches:
                if idx < len(approach.asr_cumulative):
                    payload[f"asr/{approach.label}"] = approach.asr_cumulative[idx]
                if idx < len(approach.margin_mean):
                    payload[f"margin_mean/{approach.label}"] = approach.margin_mean[idx]
                if idx < len(approach.saliency_mean):
                    payload[f"saliency_mean/{approach.label}"] = approach.saliency_mean[idx]
            run.log(payload)

    run.finish()


def main():
    args = parse_args()

    for root in args.input_roots:
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"input root not found: {root}")

    if args.watch and args.interval <= 0:
        raise ValueError("--interval must be > 0 when --watch is enabled")

    scan_index = 0
    while True:
        model_to_curves = collect_all_models(args.input_roots)
        if not model_to_curves:
            raise ValueError("No valid approach data found in given input roots")

        print(f"[INFO] models discovered: {', '.join(sorted(model_to_curves.keys()))}")

        for model_name in sorted(model_to_curves.keys()):
            eps_map = model_to_curves[model_name]
            for eps in sorted(eps_map.keys(), key=_eps_sort_key):
                approaches = eps_map[eps]
                print(
                    f"[INFO] logging dashboard for model={model_name} eps={eps} "
                    f"approaches={len(approaches)}"
                )
                log_model_dashboard(args, model_name, eps, approaches, scan_index=scan_index)

        if not args.watch:
            print("[DONE] W&B dashboard logging completed")
            break

        scan_index += 1
        print(f"[INFO] waiting {args.interval}s before next refresh")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
