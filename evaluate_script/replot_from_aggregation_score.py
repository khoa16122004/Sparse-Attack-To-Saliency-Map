import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _read_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _safe_token(name: str) -> str:
    cleaned = name.strip() or "item"
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in cleaned)


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _approach_sort_key(name: str) -> Tuple[int, str]:
    order = [
        "Clean",
        "Saliency-Optimized",
        "Delete-Oriented",
        "Insert-Oriented",
        "Del-Ins Optimized",
        "Saliency-Del-Ins Optimized",
    ]
    try:
        idx = order.index(name)
    except ValueError:
        idx = len(order)
    return idx, name.lower()


def _approach_color_map(approaches: Sequence[str]) -> Dict[str, str]:
    fixed = {
        "Clean": "#117733",
        "Saliency-Optimized": "#D55E00",
        "Delete-Oriented": "#0072B2",
        "Insert-Oriented": "#E69F00",
        "Del-Ins Optimized": "#332288",
        "Saliency-Del-Ins Optimized": "#CC79A7",
    }
    out: Dict[str, str] = {}
    palette = [
        "#4C72B0",
        "#55A868",
        "#C44E52",
        "#8172B2",
        "#CCB974",
        "#64B5CD",
        "#8C8C8C",
    ]
    unknown = [a for a in sorted(set(approaches), key=_approach_sort_key) if a not in fixed]
    for name in approaches:
        if name in fixed:
            out[name] = fixed[name]
    for idx, name in enumerate(unknown):
        out[name] = palette[idx % len(palette)]
    return out


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#666666")


def _auto_ylim(values: Sequence[float], pad_ratio: float = 0.08) -> Optional[Tuple[float, float]]:
    finite = [float(v) for v in values if not math.isnan(float(v)) and not math.isinf(float(v))]
    if not finite:
        return None
    lo = min(finite)
    hi = max(finite)
    if hi <= lo:
        delta = max(0.01, abs(lo) * 0.03)
        lo -= delta
        hi += delta
    else:
        span = hi - lo
        pad = max(1e-6, span * pad_ratio)
        lo -= pad
        hi += pad
    return lo, hi


def _discover_method_dirs(aggregation_root: Path, folder_pattern: str, include_regex: Optional[str]) -> List[Path]:
    candidates = [p for p in sorted(aggregation_root.glob(folder_pattern)) if p.is_dir()]
    if include_regex:
        regex = re.compile(include_regex)
        candidates = [p for p in candidates if regex.search(p.name)]

    out: List[Path] = []
    for folder in candidates:
        marker = folder / "_method_aggregates" / "summary_ins_del_imd_spearman_by_approach.json"
        if marker.exists():
            out.append(folder)
    return out


def _method_label(folder_name: str) -> str:
    if "__" in folder_name:
        return folder_name.split("__", 1)[1]
    return folder_name


def _extract_metric_table(method_dirs: Sequence[Path], metrics: Sequence[str]) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], List[str]]:
    # method -> approach -> metric -> value
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_approaches: List[str] = []

    for method_dir in method_dirs:
        method_name = _method_label(method_dir.name)
        path = method_dir / "_method_aggregates" / "summary_ins_del_imd_spearman_by_approach.json"
        payload = _read_json(path)
        if payload is None:
            continue
        approaches = payload.get("approaches")
        if not isinstance(approaches, dict):
            continue

        method_block: Dict[str, Dict[str, float]] = {}
        for approach_name, stats in approaches.items():
            if not isinstance(approach_name, str) or not isinstance(stats, dict):
                continue
            metric_block: Dict[str, float] = {}
            for metric in metrics:
                value = _as_float(stats.get(metric))
                if value is not None:
                    metric_block[metric] = value
            if metric_block:
                method_block[approach_name] = metric_block
                all_approaches.append(approach_name)

        if method_block:
            table[method_name] = method_block

    sorted_approaches = sorted(set(all_approaches), key=_approach_sort_key)
    return table, sorted_approaches


def _rank_approaches(
    table: Dict[str, Dict[str, Dict[str, float]]],
    approaches: Sequence[str],
    metric: str,
) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    for approach in approaches:
        values: List[float] = []
        for method_block in table.values():
            if approach in method_block and metric in method_block[approach]:
                values.append(float(method_block[approach][metric]))
        if values:
            rows.append((approach, float(np.mean(values))))
    return rows


def _plot_approach_ranking(
    rows: Sequence[Tuple[str, float]],
    title: str,
    y_label: str,
    output_path: Path,
    dpi: int,
    descending: bool,
) -> None:
    if not rows:
        return
    ordered = sorted(rows, key=lambda item: item[1], reverse=descending)
    labels = [name for name, _ in ordered]
    values = [value for _, value in ordered]
    colors = _approach_color_map(labels)

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 6.2))
    _style_axes(ax)

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=[colors[l] for l in labels], edgecolor="#2f2f2f", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    y_range = _auto_ylim(values, pad_ratio=0.12)
    if y_range is not None:
        ax.set_ylim(y_range[0], y_range[1])

    for rect, v in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() * 0.5, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_grouped_by_method(
    table: Dict[str, Dict[str, Dict[str, float]]],
    approaches: Sequence[str],
    metric: str,
    output_path: Path,
    dpi: int,
    sort_methods: str,
) -> None:
    if not table or not approaches:
        return

    methods = list(table.keys())
    if sort_methods == "alphabetical":
        methods = sorted(methods)
    elif sort_methods == "metric-mean":
        methods = sorted(
            methods,
            key=lambda m: float(np.mean([table[m][a][metric] for a in approaches if a in table[m] and metric in table[m][a]])
                                if any(a in table[m] and metric in table[m][a] for a in approaches) else -1e9),
            reverse=True,
        )

    colors = _approach_color_map(approaches)
    fig, ax = plt.subplots(1, 1, figsize=(max(12.0, len(methods) * 1.7), 6.8))
    _style_axes(ax)

    x = np.arange(len(methods), dtype=float)
    n = max(1, len(approaches))
    width = 0.82 / n

    all_values: List[float] = []
    for idx, approach in enumerate(approaches):
        y = []
        for m in methods:
            value = None
            if approach in table[m]:
                value = table[m][approach].get(metric)
            y.append(np.nan if value is None else float(value))
        all_values.extend([float(v) for v in y if not np.isnan(v)])
        shift = (idx - (n - 1) / 2.0) * width
        ax.bar(x + shift, y, width=width, label=approach, color=colors.get(approach, "#4C72B0"), alpha=0.92, linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=12, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by Method and Approach")
    ax.legend(loc="best", ncol=min(3, len(approaches)))

    y_range = _auto_ylim(all_values, pad_ratio=0.12)
    if y_range is not None:
        ax.set_ylim(y_range[0], y_range[1])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _load_curve_data(method_dirs: Sequence[Path]) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    # method -> approach -> {deletion, insertion}
    out: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for method_dir in method_dirs:
        method_name = _method_label(method_dir.name)
        path = method_dir / "_method_aggregates" / "full_avg_compare_approaches_curves.json"
        payload = _read_json(path)
        if payload is None:
            continue
        approaches = payload.get("approaches")
        if not isinstance(approaches, dict):
            continue

        method_block: Dict[str, Dict[str, List[float]]] = {}
        for approach_name, curve_block in approaches.items():
            if not isinstance(approach_name, str) or not isinstance(curve_block, dict):
                continue
            d_raw = curve_block.get("deletion")
            i_raw = curve_block.get("insertion")
            if not isinstance(d_raw, list) or not isinstance(i_raw, list):
                continue
            d_vals = [float(v) for v in d_raw if _as_float(v) is not None]
            i_vals = [float(v) for v in i_raw if _as_float(v) is not None]
            if d_vals or i_vals:
                method_block[approach_name] = {"deletion": d_vals, "insertion": i_vals}

        if method_block:
            out[method_name] = method_block
    return out


def _mean_curve(curves: Sequence[Sequence[float]]) -> List[float]:
    valid = [list(c) for c in curves if c]
    if not valid:
        return []
    max_len = max(len(c) for c in valid)
    out: List[float] = []
    for idx in range(max_len):
        samples = [curve[idx] if idx < len(curve) else curve[-1] for curve in valid]
        out.append(float(np.mean(samples)))
    return out


def _plot_combined_curves(
    curve_table: Dict[str, Dict[str, Dict[str, List[float]]]],
    approaches: Sequence[str],
    curve_name: str,
    output_path: Path,
    dpi: int,
    top_k: int,
) -> None:
    if not curve_table:
        return

    ranked_rows = []
    for approach in approaches:
        last_vals: List[float] = []
        for method_block in curve_table.values():
            block = method_block.get(approach)
            if not block:
                continue
            curve = block.get(curve_name, [])
            if curve:
                last_vals.append(float(curve[-1]))
        if last_vals:
            ranked_rows.append((approach, float(np.mean(last_vals))))

    ranked_rows.sort(key=lambda item: item[1], reverse=True)
    selected = [name for name, _ in ranked_rows]
    if top_k > 0:
        selected = selected[:top_k]

    colors = _approach_color_map(selected)
    fig, ax = plt.subplots(1, 1, figsize=(11.8, 6.4))
    _style_axes(ax)

    y_pool: List[float] = []
    for approach in selected:
        curves = []
        for method_block in curve_table.values():
            block = method_block.get(approach)
            if not block:
                continue
            curve = block.get(curve_name, [])
            if curve:
                curves.append(curve)
        mean_curve = _mean_curve(curves)
        if not mean_curve:
            continue
        x = np.linspace(0.0, 1.0, num=len(mean_curve), endpoint=True)
        y = np.asarray(mean_curve, dtype=float)
        y_pool.extend(float(v) for v in y)
        ax.plot(x, y, label=approach, color=colors.get(approach, "#4C72B0"), linewidth=2.4)

    if not y_pool:
        plt.close(fig)
        return

    ax.set_xlim(0.0, 1.0)
    y_range = _auto_ylim(y_pool, pad_ratio=0.08)
    if y_range is not None:
        ax.set_ylim(y_range[0], y_range[1])
    ax.set_xlabel("Perturbation ratio")
    ax.set_ylabel(f"{curve_name} score")
    ax.set_title(f"Average {curve_name.title()} Curves Across Methods")
    ax.legend(loc="best", ncol=1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-plot from already aggregated outputs inside aggration_score without raw summary files."
    )
    parser.add_argument("--aggregation-root", type=str, required=True, help="Path to aggration_score root folder.")
    parser.add_argument("--folder-pattern", type=str, default="*", help="Glob for selecting method folders under root.")
    parser.add_argument("--include-regex", type=str, default=None, help="Optional regex filter on method folder names.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output folder. Default: <aggregation-root>/replot.")
    parser.add_argument("--metrics", nargs="+", default=["del", "ins", "imd", "spearman"], help="Metrics to re-plot.")
    parser.add_argument("--sort-methods", choices=["none", "alphabetical", "metric-mean"], default="alphabetical")
    parser.add_argument("--top-k-approaches", type=int, default=0, help="Keep only top-k approaches for curve overlays. 0 means all.")
    parser.add_argument("--descending", action="store_true", help="Sort ranking bars high-to-low. Default low-to-high.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png"], help="Output formats, e.g. png pdf.")
    args = parser.parse_args()

    aggregation_root = Path(args.aggregation_root).resolve()
    if not aggregation_root.exists() or not aggregation_root.is_dir():
        raise FileNotFoundError(f"aggregation root not found: {aggregation_root}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (aggregation_root / "replot")
    out_dir.mkdir(parents=True, exist_ok=True)

    method_dirs = _discover_method_dirs(aggregation_root, args.folder_pattern, args.include_regex)
    if not method_dirs:
        raise RuntimeError("No valid aggregated method folders found.")

    metrics = [m.strip() for m in args.metrics if m.strip()]
    table, approaches = _extract_metric_table(method_dirs, metrics)
    if not table or not approaches:
        raise RuntimeError("No metric data found in selected folders.")

    print(f"[INFO] discovered method folders: {len(method_dirs)}")
    print(f"[INFO] methods with metric data: {len(table)}")
    print(f"[INFO] approaches: {', '.join(approaches)}")

    for metric in metrics:
        ranking_rows = _rank_approaches(table, approaches, metric)
        for ext in args.formats:
            rank_out = out_dir / f"approach_ranking_{_safe_token(metric)}.{ext}"
            grouped_out = out_dir / f"method_grouped_{_safe_token(metric)}.{ext}"
            _plot_approach_ranking(
                rows=ranking_rows,
                title=f"Approach Ranking by {metric}",
                y_label=metric,
                output_path=rank_out,
                dpi=args.dpi,
                descending=bool(args.descending),
            )
            _plot_grouped_by_method(
                table=table,
                approaches=approaches,
                metric=metric,
                output_path=grouped_out,
                dpi=args.dpi,
                sort_methods=args.sort_methods,
            )

    curve_table = _load_curve_data(method_dirs)
    for curve_name in ("deletion", "insertion"):
        for ext in args.formats:
            curve_out = out_dir / f"curves_over_methods_{curve_name}.{ext}"
            _plot_combined_curves(
                curve_table=curve_table,
                approaches=approaches,
                curve_name=curve_name,
                output_path=curve_out,
                dpi=args.dpi,
                top_k=int(args.top_k_approaches),
            )

    print("=== Done ===")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
