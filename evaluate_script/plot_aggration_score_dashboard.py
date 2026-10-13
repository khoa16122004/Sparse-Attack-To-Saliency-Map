import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.titleweight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["font.size"] = 28
plt.rcParams["axes.labelsize"] = 28
plt.rcParams["xtick.labelsize"] = 29
plt.rcParams["ytick.labelsize"] = 28
plt.rcParams["legend.fontsize"] = 29


APPROACH_ORDER = [
    "Clean",
    "Saliency-Optimized",
    "Delete-Oriented",
    "Insert-Oriented",
    "Del-Ins Optimized",
    "Saliency-Del-Ins Optimized",
]

APPROACH_LABELS = {
    "Clean": "Clean",
    "Saliency-Optimized": "Shift-Only",
    "Delete-Oriented": "Deletion-Driven",
    "Insert-Oriented": "Insertion-Driven",
    "Del-Ins Optimized": "Faithfulness-Driven",
    "Saliency-Del-Ins Optimized": "Shift-Faithfulness",
}

# Reverse lookup used to recover the canonical ordering / color for a
# translated display label (needed because legend handles are keyed by the
# *display* label, not the raw approach name).
LABEL_TO_APPROACH = {v: k for k, v in APPROACH_LABELS.items()}


def _read_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


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
    try:
        idx = APPROACH_ORDER.index(name)
    except ValueError:
        idx = len(APPROACH_ORDER)
    return idx, name.lower()


def _label_sort_key(label: str) -> Tuple[int, str]:
    # Legends are built from display labels (e.g. "Shift-Only"), so map back
    # to the raw approach name before applying the canonical ordering.
    approach = LABEL_TO_APPROACH.get(label, label)
    return _approach_sort_key(approach)


def _approach_color_map(approaches: Sequence[str]) -> Dict[str, str]:
    fixed = {
        "Clean": "#117733",
        "Saliency-Optimized": "#D55E00",
        "Delete-Oriented": "#0072B2",
        "Insert-Oriented": "#E69F00",
        "Del-Ins Optimized": "#332288",
        # Shift-Faithfulness must always render in red.
        "Saliency-Del-Ins Optimized": "#D62728",
    }
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#8C8C8C"]

    out: Dict[str, str] = {}
    unknown: List[str] = []
    for name in sorted(set(approaches), key=_approach_sort_key):
        if name in fixed:
            out[name] = fixed[name]
        else:
            unknown.append(name)

    for i, name in enumerate(unknown):
        out[name] = palette[i % len(palette)]
    return out


def _mean_curve_with_last_padding(curves: Sequence[Sequence[float]]) -> List[float]:
    valid = [list(c) for c in curves if c]
    if not valid:
        return []
    max_len = max(len(c) for c in valid)
    out: List[float] = []
    for idx in range(max_len):
        vals = [curve[idx] if idx < len(curve) else curve[-1] for curve in valid]
        out.append(float(np.mean(vals)))
    return out


def _auto_ylim(values: Sequence[float], pad_ratio: float, clamp_lo: Optional[float], clamp_hi: Optional[float]) -> Optional[Tuple[float, float]]:
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
        pad = max(1e-6, span * max(0.0, pad_ratio))
        lo -= pad
        hi += pad

    if clamp_lo is not None:
        lo = max(lo, float(clamp_lo))
    if clamp_hi is not None:
        hi = min(hi, float(clamp_hi))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _major_ticks(start: float, end: float, count: int = 5) -> np.ndarray:
    n = max(2, int(count))
    return np.linspace(float(start), float(end), n)


def _markevery_indices(length: int, count: int) -> Optional[List[int]]:
    if count <= 0 or length <= 0:
        return None
    if count >= length:
        return list(range(length))
    step = max(1, (length - 1) // max(1, count - 1))
    idxs = list(range(0, length, step))
    if idxs[-1] != (length - 1):
        idxs.append(length - 1)
    if len(idxs) > count:
        idxs = idxs[: count - 1] + [length - 1]
    return idxs


def _sample_line_points(x: np.ndarray, y: np.ndarray, point_count: int) -> Tuple[np.ndarray, np.ndarray]:
    if point_count <= 0:
        return x, y
    idxs = _markevery_indices(int(len(x)), int(point_count))
    if idxs is None:
        return x, y
    return x[idxs], y[idxs]


def _apply_two_decimal_ticks(ax: plt.Axes, apply_x: bool, apply_y: bool) -> None:
    if apply_x:
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if apply_y:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def _iteration_ticks(
    x_lo: float,
    x_hi: float,
    tick_count: int,
    mode: str,
    fixed_ticks: Sequence[int],
) -> List[int]:
    if x_hi <= x_lo:
        return [int(round(x_lo))]

    if mode == "fixed":
        ticks = [int(t) for t in fixed_ticks if x_lo <= int(t) <= x_hi]
        if len(ticks) >= 2:
            return sorted(set(ticks))

    # Default: evenly spaced ticks for visually uniform spacing.
    fallback = [int(round(v)) for v in _major_ticks(x_lo, x_hi, tick_count)]
    uniq = sorted(set(fallback))
    if len(uniq) < 2:
        uniq = [int(round(x_lo)), int(round(x_hi))]
    return uniq


def _parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        out.append(value)
    return out


def _style_axes(ax: plt.Axes, grid_alpha: float) -> None:
    ax.set_facecolor("#ffffff")
    _ = grid_alpha
    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_color("#666666")
    ax.tick_params(axis="both", labelsize=22, width=1.4, length=5.0)


def _shrink_axis_box(ax: plt.Axes, width_scale: float, height_scale: float) -> None:
    w = min(max(float(width_scale), 0.35), 1.0)
    h = min(max(float(height_scale), 0.35), 1.0)
    box = ax.get_position()
    new_w = box.width * w
    new_h = box.height * h
    new_x = box.x0 + (box.width - new_w) * 0.5
    new_y = box.y0 + (box.height - new_h) * 0.5
    ax.set_position([new_x, new_y, new_w, new_h])


def _safe_name(name: str) -> str:
    s = name.strip() or "plot"
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in s)


def _discover_method_dirs(root: Path, folder_pattern: str, include_regex: Optional[str]) -> List[Path]:
    candidates = [p for p in sorted(root.glob(folder_pattern)) if p.is_dir()]
    if include_regex:
        regex = re.compile(include_regex)
        candidates = [p for p in candidates if regex.search(p.name)]

    out: List[Path] = []
    for folder in candidates:
        marker = folder / "_method_aggregates" / "summary_ins_del_imd_spearman_by_approach.json"
        if marker.exists():
            out.append(folder)
    return out


def _method_name(folder: Path) -> str:
    if "__" in folder.name:
        return folder.name.split("__", 1)[1]
    return folder.name


def _load_curves_from_aggregates(method_dirs: Sequence[Path]) -> Tuple[Dict[str, Dict[str, List[float]]], List[str]]:
    # approach -> {deletion: [curve per method], insertion: [curve per method]}
    pooled: Dict[str, Dict[str, List[List[float]]]] = {}

    for folder in method_dirs:
        path = folder / "_method_aggregates" / "full_avg_compare_approaches_curves.json"
        payload = _read_json(path)
        if payload is None:
            continue
        approaches_block = payload.get("approaches")
        if not isinstance(approaches_block, dict):
            continue

        for approach, block in approaches_block.items():
            if not isinstance(approach, str) or not isinstance(block, dict):
                continue
            del_raw = block.get("deletion")
            ins_raw = block.get("insertion")
            if not isinstance(del_raw, list) or not isinstance(ins_raw, list):
                continue

            del_curve = [float(v) for v in del_raw if _as_float(v) is not None]
            ins_curve = [float(v) for v in ins_raw if _as_float(v) is not None]
            if not del_curve and not ins_curve:
                continue

            holder = pooled.setdefault(approach, {"deletion": [], "insertion": []})
            if del_curve:
                holder["deletion"].append(del_curve)
            if ins_curve:
                holder["insertion"].append(ins_curve)

    mean_curves: Dict[str, Dict[str, List[float]]] = {}
    for approach, block in pooled.items():
        mean_curves[approach] = {
            "deletion": _mean_curve_with_last_padding(block.get("deletion", [])),
            "insertion": _mean_curve_with_last_padding(block.get("insertion", [])),
        }

    approaches = sorted(mean_curves.keys(), key=_approach_sort_key)
    return mean_curves, approaches


def _decode_tuple_key_records(payload: object, key_len: int) -> Dict[Tuple[str, ...], object]:
    out: Dict[Tuple[str, ...], object] = {}
    if not isinstance(payload, list):
        return out
    for item in payload:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not isinstance(key, list) or len(key) != key_len:
            continue
        if not all(isinstance(k, str) for k in key):
            continue
        out[tuple(key)] = item.get("value")
    return out


def _load_objective_iteration_curves(method_dirs: Sequence[Path]) -> Tuple[Dict[str, Dict[str, List[float]]], List[str]]:
    # approach -> metric -> list[curve], then mean
    pooled: Dict[str, Dict[str, List[List[float]]]] = {}

    for folder in method_dirs:
        cache_path = folder / "_method_aggregates" / "reproduce_cache.json"
        payload = _read_json(cache_path)
        if payload is None:
            continue
        data = payload.get("data")
        if not isinstance(data, dict):
            continue

        obj_records = _decode_tuple_key_records(data.get("per_run_objective_mean"), key_len=3)
        sp_records = _decode_tuple_key_records(data.get("per_run_spearman_mean"), key_len=3)

        for run_key, obj_block in obj_records.items():
            if not isinstance(obj_block, dict):
                continue
            approach = run_key[2]
            holder = pooled.setdefault(approach, {"deletion_obj": [], "insertion_obj": [], "spearman": []})

            del_curve_raw = obj_block.get("deletion_obj")
            ins_curve_raw = obj_block.get("insertion_obj")
            if isinstance(del_curve_raw, list):
                del_curve = [float(v) for v in del_curve_raw if _as_float(v) is not None]
                if del_curve:
                    holder["deletion_obj"].append(del_curve)
            if isinstance(ins_curve_raw, list):
                ins_curve = [float(v) for v in ins_curve_raw if _as_float(v) is not None]
                if ins_curve:
                    holder["insertion_obj"].append(ins_curve)

        for run_key, sp_curve_raw in sp_records.items():
            if not isinstance(sp_curve_raw, list):
                continue
            approach = run_key[2]
            holder = pooled.setdefault(approach, {"deletion_obj": [], "insertion_obj": [], "spearman": []})
            sp_curve = [float(v) for v in sp_curve_raw if _as_float(v) is not None]
            if sp_curve:
                holder["spearman"].append(sp_curve)

    mean_curves: Dict[str, Dict[str, List[float]]] = {}
    for approach, block in pooled.items():
        mean_curves[approach] = {
            "deletion_obj": _mean_curve_with_last_padding(block.get("deletion_obj", [])),
            "insertion_obj": _mean_curve_with_last_padding(block.get("insertion_obj", [])),
            "spearman": _mean_curve_with_last_padding(block.get("spearman", [])),
        }

    approaches = sorted(mean_curves.keys(), key=_approach_sort_key)
    return mean_curves, approaches


def _load_spearman_scores(method_dirs: Sequence[Path]) -> Tuple[Dict[str, float], List[str]]:
    # average full spearman per approach across all selected methods
    pooled: Dict[str, List[float]] = {}

    for folder in method_dirs:
        path = folder / "_method_aggregates" / "spearman_comparison" / "spearman_compare_approaches_summary.json"
        payload = _read_json(path)
        if payload is None:
            continue
        full = payload.get("full")
        if not isinstance(full, dict):
            continue

        for approach, value in full.items():
            if not isinstance(approach, str):
                continue
            num = _as_float(value)
            if num is None:
                continue
            pooled.setdefault(approach, []).append(num)

    mean_scores = {k: float(np.mean(v)) for k, v in pooled.items() if v}
    approaches = sorted(mean_scores.keys(), key=_approach_sort_key)
    return mean_scores, approaches


def _maybe_slice_curve(curve: Sequence[float], start: int, end: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if not curve:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    lo = max(0, int(start))
    hi = len(curve) - 1 if end is None else min(len(curve) - 1, int(end))
    if hi < lo:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x = np.arange(lo, hi + 1, dtype=float)
    y = np.asarray(curve[lo:hi + 1], dtype=float)
    return x, y


def _build_ordered_legend(fig: plt.Figure, axes: Sequence[plt.Axes]) -> None:
    """Collect handles/labels from all axes and render one legend ordered by
    the canonical approach order (mapped through the display label)."""
    legend_items: Dict[str, object] = {}
    for ax in axes:
        handles_ax, labels_ax = ax.get_legend_handles_labels()
        for handle, label in zip(handles_ax, labels_ax):
            legend_items[label] = handle

    labels = sorted(legend_items.keys(), key=_label_sort_key)
    handles = [legend_items[k] for k in labels]
    if labels:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=max(1, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
            fontsize=24,
            borderpad=0.25,
            handletextpad=0.5,
            columnspacing=0.9,
        )


def _plot_dual_faithfulness(
    mean_curves: Dict[str, Dict[str, List[float]]],
    approaches: Sequence[str],
    output_path: Path,
    dpi: int,
    x_min: float,
    x_max: float,
    y_min: Optional[float],
    y_max: Optional[float],
    line_width: float,
    line_alpha: float,
    marker_size: float,
    grid_alpha: float,
    legend_cols: int,
    auto_y_pad: float,
    tick_count: int,
    square_subplots: bool,
    subplot_width_scale: float,
    subplot_height_scale: float,
    subplot_wspace: float,
    marker_count: int,
    curve_point_count: int,
) -> None:
    if not mean_curves:
        return

    colors = _approach_color_map(approaches)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), sharex=True)
    panels = [("deletion", "Deletion"), ("insertion", "Insertion")]

    for ax, (key, title) in zip(axes, panels):
        _style_axes(ax, grid_alpha=grid_alpha)
        ax.set_title(
            title,
            fontsize=28,
            fontweight="normal",
            pad=14,
        )
        y_pool: List[float] = []

        # Draw approaches in the canonical order so overlapping lines/markers
        # are layered consistently with the legend order.
        for approach in sorted(approaches, key=_approach_sort_key):
            curve = mean_curves.get(approach, {}).get(key, [])
            if not curve:
                continue
            x = np.linspace(0.0, 1.0, num=len(curve), endpoint=True)
            y = np.asarray(curve, dtype=float)
            mask = (x >= x_min) & (x <= x_max)
            if not np.any(mask):
                continue
            x_plot = x[mask]
            y_plot = y[mask]
            x_draw, y_draw = _sample_line_points(x_plot, y_plot, curve_point_count)
            y_pool.extend(float(v) for v in y_plot)

            ax.plot(
                x_draw,
                y_draw,
                label=APPROACH_LABELS.get(approach, approach),
                color=colors.get(approach, "#4C72B0"),
                linewidth=line_width,
                alpha=line_alpha,
                marker="o",
                markersize=marker_size,
                markerfacecolor="white",
                markeredgecolor=colors.get(approach, "#4C72B0"),
                markeredgewidth=max(1.3, line_width * 0.45),
                markevery=_markevery_indices(len(x_draw), marker_count),
            )

        ax.set_xlim(float(x_min), float(x_max))
        ax.set_xticks(_major_ticks(x_min, x_max, tick_count))
        if y_min is not None and y_max is not None and y_max > y_min:
            ax.set_ylim(float(y_min), float(y_max))
        else:
            y_lim = _auto_ylim(y_pool, pad_ratio=auto_y_pad, clamp_lo=-1.0, clamp_hi=1.0)
            if y_lim is not None:
                ax.set_ylim(y_lim[0], y_lim[1])
        y_lo, y_hi = ax.get_ylim()
        ax.set_yticks(_major_ticks(y_lo, y_hi, tick_count))
        _apply_two_decimal_ticks(ax, apply_x=True, apply_y=True)
        if square_subplots:
            ax.set_box_aspect(1.0)
        _shrink_axis_box(ax, subplot_width_scale, subplot_height_scale)

        ax.set_xlabel("Step")
        ax.set_ylabel("Score")

    _build_ordered_legend(fig, axes)

    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.98), w_pad=0.35)
    fig.subplots_adjust(wspace=max(0.0, float(subplot_wspace)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_objective_triptych(
    objective_curves: Dict[str, Dict[str, List[float]]],
    approaches: Sequence[str],
    output_path: Path,
    dpi: int,
    iter_start: int,
    iter_end: Optional[int],
    obj_y_min: Optional[float],
    obj_y_max: Optional[float],
    line_width: float,
    line_alpha: float,
    marker_size: float,
    grid_alpha: float,
    legend_cols: int,
    auto_y_pad: float,
    tick_count: int,
    iter_tick_mode: str,
    iter_fixed_ticks: Sequence[int],
    square_subplots: bool,
    subplot_width_scale: float,
    subplot_height_scale: float,
    subplot_wspace: float,
    marker_count: int,
    curve_point_count: int,
) -> None:
    if not objective_curves:
        return

    colors = _approach_color_map(approaches)
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.3), sharex=False)
    panels = [
        ("deletion_obj", r"$\mathcal{L}_{del}$"),
        ("insertion_obj", r"$\mathcal{L}_{ins}$"),
        ("spearman", r"$\mathcal{L}_{saliency}$"),
    ]

    for ax, (metric_key, _title) in zip(axes, panels):
        ax.set_title(
            _title,
            fontsize=28,
            fontweight="normal",
            pad=14,
        )
        _style_axes(ax, grid_alpha=grid_alpha)
        y_pool: List[float] = []
        x_observed_max: Optional[int] = None

        # Draw approaches in the canonical order for consistent layering.
        for approach in sorted(approaches, key=_approach_sort_key):
            curve = objective_curves.get(approach, {}).get(metric_key, [])
            x, y = _maybe_slice_curve(curve, iter_start, iter_end)
            if x.size == 0:
                continue
            x_draw, y_draw = _sample_line_points(x, y, curve_point_count)
            y_pool.extend(float(v) for v in y)
            x_observed_max = int(x[-1]) if x_observed_max is None else max(x_observed_max, int(x[-1]))
            ax.plot(
                x_draw,
                y_draw,
                label=APPROACH_LABELS.get(approach, approach),
                color=colors.get(approach, "#4C72B0"),
                linewidth=line_width,
                alpha=line_alpha,
                marker="o",
                markersize=marker_size,
                markerfacecolor="white",
                markeredgecolor=colors.get(approach, "#4C72B0"),
                markeredgewidth=max(1.3, line_width * 0.45),
                markevery=_markevery_indices(len(x_draw), marker_count),
            )

        if obj_y_min is not None and obj_y_max is not None and obj_y_max > obj_y_min:
            ax.set_ylim(float(obj_y_min), float(obj_y_max))
        else:
            y_lim = _auto_ylim(y_pool, pad_ratio=auto_y_pad, clamp_lo=None, clamp_hi=None)
            if y_lim is not None:
                ax.set_ylim(y_lim[0], y_lim[1])

        if x_observed_max is not None:
            x_lo = max(1, int(iter_start))
            x_hi = int(x_observed_max) if iter_end is None else min(int(x_observed_max), int(iter_end))
            if x_hi < x_lo:
                x_hi = x_lo
            ax.set_xlim(float(x_lo), float(x_hi))
        else:
            x_lo, x_hi = ax.get_xlim()

        x_ticks = _iteration_ticks(x_lo, x_hi, tick_count, iter_tick_mode, iter_fixed_ticks)
        if len(x_ticks) >= 2:
            ax.set_xticks(x_ticks)

        y_lo, y_hi = ax.get_ylim()
        ax.set_yticks(_major_ticks(y_lo, y_hi, tick_count))
        _apply_two_decimal_ticks(ax, apply_x=False, apply_y=True)
        if square_subplots:
            ax.set_box_aspect(1.0)
        _shrink_axis_box(ax, subplot_width_scale, subplot_height_scale)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")

    _build_ordered_legend(fig, axes)

    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.98), w_pad=0.35)
    fig.subplots_adjust(wspace=max(0.0, float(subplot_wspace)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_spearman_bar(
    spearman_scores: Dict[str, float],
    approaches: Sequence[str],
    output_path: Path,
    dpi: int,
    bar_alpha: float,
    bar_line_width: float,
    grid_alpha: float,
    descending: bool,
    bar_y_min: Optional[float],
    bar_y_max: Optional[float],
    auto_y_pad: float,
    tick_count: int,
) -> None:
    if not spearman_scores:
        return

    # Build the color map from the RAW approach names (same names used by the
    # line-chart figures) so bar colors always match figures 1 and 2,
    # regardless of how the bars end up sorted by value.
    colors_by_approach = _approach_color_map(approaches)

    rows = [(a, spearman_scores[a]) for a in approaches if a in spearman_scores]
    rows = sorted(rows, key=lambda item: item[1], reverse=bool(descending))
    labels = [APPROACH_LABELS.get(a, a) for a, _ in rows]
    values = [v for _, v in rows]
    bar_colors = [colors_by_approach.get(a, "#4C72B0") for a, _ in rows]

    fig, ax = plt.subplots(1, 1, figsize=(10.8, 5.2))
    _style_axes(ax, grid_alpha=grid_alpha)

    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        values,
        color=bar_colors,
        edgecolor="#2f2f2f",
        linewidth=bar_line_width,
        alpha=bar_alpha,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.set_ylabel("Mean Spearman")

    if values:
        # Start from data-driven limits, then apply optional user clamps.
        data_min = float(min(values))
        data_max = float(max(values))
        y_lo = min(0.0, data_min)
        y_hi = max(1.0, data_max + max(0.01, abs(data_max) * auto_y_pad))

        if bar_y_min is not None:
            y_lo = float(bar_y_min)
        if bar_y_max is not None:
            y_hi = float(bar_y_max)

        # If a clamp would hide bars, relax the lower bound automatically.
        if y_lo > data_min:
            y_lo = data_min - max(0.01, abs(data_min) * 0.02)
        if y_hi <= y_lo:
            y_hi = y_lo + 1e-6

        ax.set_ylim(y_lo, y_hi)

    y_lo, y_hi = ax.get_ylim()
    ax.set_yticks(_major_ticks(y_lo, y_hi, tick_count))
    _apply_two_decimal_ticks(ax, apply_x=False, apply_y=True)
    label_offset = max(0.003, (y_hi - y_lo) * 0.006)

    for rect, value in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() * 0.5,
            value + label_offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=25,
            fontweight="normal",
            color="#222222",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot dashboard directly from aggration_score outputs."
    )
    parser.add_argument("--aggregation-root", type=str, default="aggration_score", help="Path to aggration_score root.")
    parser.add_argument("--folder-pattern", type=str, default="*", help="Glob to pick method folders under root.")
    parser.add_argument("--include-regex", type=str, default=None, help="Optional regex filter on method folder names.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output folder. Default: <aggregation-root>/dashboard_plots.")
    parser.add_argument("--formats", nargs="+", default=["png"], help="Output formats, e.g. png pdf.")
    parser.add_argument("--dpi", type=int, default=320)

    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--iter-start", type=int, default=1)
    parser.add_argument("--iter-end", type=int, default=-1)
    parser.add_argument("--obj-y-min", type=float, default=None)
    parser.add_argument("--obj-y-max", type=float, default=None)
    parser.add_argument("--bar-y-min", type=float, default=None)
    parser.add_argument("--bar-y-max", type=float, default=0.95)
    parser.add_argument("--auto-y-pad", type=float, default=0.08)

    parser.add_argument("--line-width", type=float, default=3.2)
    parser.add_argument("--line-alpha", type=float, default=1.0)
    parser.add_argument("--marker-size", type=float, default=4.8)
    parser.add_argument("--bar-alpha", type=float, default=1.0)
    parser.add_argument("--bar-line-width", type=float, default=1.6)
    parser.add_argument("--grid-alpha", type=float, default=0.35)
    parser.add_argument("--legend-cols", type=int, default=3)
    parser.add_argument("--tick-count", type=int, default=5, help="Target number of major ticks for numeric axes.")
    parser.add_argument("--marker-count", type=int, default=14, help="Approximate number of visible markers per line (<=0 disables markers spacing control).")
    parser.add_argument("--curve-point-count", type=int, default=0, help="How many points to keep for line connection (0 keeps all points).")
    parser.add_argument("--square-subplots", action="store_true", help="Render subplots with square box aspect for clearer zoomed view.")
    parser.add_argument("--subplot-width-scale", type=float, default=1.0, help="Shrink subplot width (0.35..1.0). Lower means narrower subplot.")
    parser.add_argument("--subplot-height-scale", type=float, default=1.0, help="Shrink subplot height (0.35..1.0). Lower means shorter subplot.")
    parser.add_argument("--dual-wspace", type=float, default=0.3, help="Horizontal spacing between the 2 subplots in figure 1.")
    parser.add_argument("--triptych-wspace", type=float, default=0.3, help="Horizontal spacing between the 3 subplots in figure 2.")
    parser.add_argument(
        "--iter-tick-mode",
        choices=["even", "fixed"],
        default="even",
        help="Iteration x-axis tick strategy: evenly spaced ticks or fixed ticks.",
    )
    parser.add_argument(
        "--iter-fixed-ticks",
        type=str,
        default="1,50,100,200",
        help="Comma-separated fixed iteration ticks, used when --iter-tick-mode=fixed.",
    )
    parser.add_argument("--descending", action="store_true", help="Sort spearman bars high-to-low.")
    parser.add_argument("--top-k-approaches", type=int, default=0, help="Keep only top-k approaches by spearman. 0 means all.")
    args = parser.parse_args()

    root = Path(args.aggregation_root).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"aggregation root not found: {root}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (root / "dashboard_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.x_max <= args.x_min:
        raise ValueError("x-max must be greater than x-min")
    if args.iter_end >= 0 and args.iter_end < args.iter_start:
        raise ValueError("iter-end must be >= iter-start, or -1")
    if int(args.tick_count) < 2:
        raise ValueError("tick-count must be >= 2")
    if int(args.marker_count) < 0:
        raise ValueError("marker-count must be >= 0")
    if int(args.curve_point_count) < 0:
        raise ValueError("curve-point-count must be >= 0")
    if float(args.subplot_width_scale) <= 0.0 or float(args.subplot_height_scale) <= 0.0:
        raise ValueError("subplot-width-scale and subplot-height-scale must be > 0")
    if float(args.dual_wspace) < 0.0 or float(args.triptych_wspace) < 0.0:
        raise ValueError("dual-wspace and triptych-wspace must be >= 0")
    iter_fixed_ticks = _parse_int_list(args.iter_fixed_ticks)

    method_dirs = _discover_method_dirs(root, args.folder_pattern, args.include_regex)
    if not method_dirs:
        raise RuntimeError("No valid method folders found under aggregation root.")

    method_names = [_method_name(p) for p in method_dirs]
    print(f"[INFO] selected method folders: {len(method_dirs)}")
    print(f"[INFO] methods: {', '.join(method_names)}")

    mean_curves, approaches_1 = _load_curves_from_aggregates(method_dirs)
    objective_curves, approaches_2 = _load_objective_iteration_curves(method_dirs)
    spearman_scores, approaches_3 = _load_spearman_scores(method_dirs)

    approaches = sorted(set(approaches_1) | set(approaches_2) | set(approaches_3), key=_approach_sort_key)
    if not approaches:
        raise RuntimeError("No approach data available for plotting.")

    if args.top_k_approaches > 0:
        ranked = sorted(
            [(a, spearman_scores.get(a, -1e9)) for a in approaches],
            key=lambda item: item[1],
            reverse=True,
        )
        top = [a for a, _ in ranked[: args.top_k_approaches]]
        approaches = [a for a in approaches if a in top]

    iter_end = None if int(args.iter_end) < 0 else int(args.iter_end)

    for ext in args.formats:
        _plot_dual_faithfulness(
            mean_curves=mean_curves,
            approaches=approaches,
            output_path=out_dir / f"01_dual_deletion_insertion.{_safe_name(ext)}",
            dpi=int(args.dpi),
            x_min=float(args.x_min),
            x_max=float(args.x_max),
            y_min=(None if args.y_min is None else float(args.y_min)),
            y_max=(None if args.y_max is None else float(args.y_max)),
            line_width=float(args.line_width),
            line_alpha=float(args.line_alpha),
            marker_size=float(args.marker_size),
            grid_alpha=float(args.grid_alpha),
            legend_cols=int(args.legend_cols),
            auto_y_pad=float(args.auto_y_pad),
            tick_count=int(args.tick_count),
            square_subplots=bool(args.square_subplots),
            subplot_width_scale=float(args.subplot_width_scale),
            subplot_height_scale=float(args.subplot_height_scale),
            subplot_wspace=float(args.dual_wspace),
            marker_count=int(args.marker_count),
            curve_point_count=int(args.curve_point_count),
        )
        _plot_objective_triptych(
            objective_curves=objective_curves,
            approaches=approaches,
            output_path=out_dir / f"02_objective_triptych.{_safe_name(ext)}",
            dpi=int(args.dpi),
            iter_start=int(args.iter_start),
            iter_end=iter_end,
            obj_y_min=(None if args.obj_y_min is None else float(args.obj_y_min)),
            obj_y_max=(None if args.obj_y_max is None else float(args.obj_y_max)),
            line_width=float(args.line_width),
            line_alpha=float(args.line_alpha),
            marker_size=float(args.marker_size),
            grid_alpha=float(args.grid_alpha),
            legend_cols=int(args.legend_cols),
            auto_y_pad=float(args.auto_y_pad),
            tick_count=int(args.tick_count),
            iter_tick_mode=str(args.iter_tick_mode),
            iter_fixed_ticks=iter_fixed_ticks,
            square_subplots=bool(args.square_subplots),
            subplot_width_scale=float(args.subplot_width_scale),
            subplot_height_scale=float(args.subplot_height_scale),
            subplot_wspace=float(args.triptych_wspace),
            marker_count=int(args.marker_count),
            curve_point_count=int(args.curve_point_count),
        )
        _plot_spearman_bar(
            spearman_scores=spearman_scores,
            approaches=approaches,
            output_path=out_dir / f"03_spearman_bar.{_safe_name(ext)}",
            dpi=int(args.dpi),
            bar_alpha=float(args.bar_alpha),
            bar_line_width=float(args.bar_line_width),
            grid_alpha=float(args.grid_alpha),
            descending=bool(args.descending),
            bar_y_min=(None if args.bar_y_min is None else float(args.bar_y_min)),
            bar_y_max=(None if args.bar_y_max is None else float(args.bar_y_max)),
            auto_y_pad=float(args.auto_y_pad),
            tick_count=int(args.tick_count),
        )

    print("=== Done ===")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()