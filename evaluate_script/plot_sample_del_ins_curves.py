import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
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


def _as_float_list(value: object) -> List[float]:
    if not isinstance(value, list):
        return []

    out: List[float] = []
    for item in value:
        try:
            num = float(item)
        except (TypeError, ValueError):
            continue
        if math.isnan(num) or math.isinf(num):
            continue
        out.append(num)
    return out


def _auc(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    arr = np.asarray(values, dtype=float)
    return float((arr.sum() - arr[0] / 2.0 - arr[-1] / 2.0) / (arr.shape[0] - 1))


def _extract_curve_block(payload: Dict[str, object]) -> Optional[Dict[str, object]]:
    for key in ("faithfulness", "causual", "causal"):
        block = payload.get(key)
        if isinstance(block, dict):
            return block
    return None


def _extract_curves(payload: Dict[str, object]) -> Optional[Dict[str, List[float]]]:
    block = _extract_curve_block(payload)
    if block is None:
        return None

    adv_del = _as_float_list(block.get("del"))
    adv_ins = _as_float_list(block.get("ins"))
    clean_del = _as_float_list(block.get("clean_del"))
    clean_ins = _as_float_list(block.get("clean_ins"))

    if not adv_del and not adv_ins and not clean_del and not clean_ins:
        return None

    return {
        "adv_del": adv_del,
        "adv_ins": adv_ins,
        "clean_del": clean_del,
        "clean_ins": clean_ins,
    }


def _extract_spearman(payload: Dict[str, object]) -> List[float]:
    return _as_float_list(payload.get("history_spearman"))


def _extract_spearman_scalar(payload: Dict[str, object]) -> Optional[float]:
    direct = payload.get("spearman_best_adv_vs_clean")
    if direct is not None:
        try:
            direct_value = float(direct)
        except (TypeError, ValueError):
            direct_value = None
        if direct_value is not None and not math.isnan(direct_value) and not math.isinf(direct_value):
            return direct_value

    history = _as_float_list(payload.get("history_spearman"))
    if history:
        return float(history[-1])
    return None


def _extract_objective_histories(payload: Dict[str, object]) -> Dict[str, List[float]]:
    return {
        "margin": _as_float_list(payload.get("history_margin")),
        "saliency": _as_float_list(payload.get("history_saliency")),
        "deletion_obj": _as_float_list(payload.get("history_del")),
        "insertion_obj": _as_float_list(payload.get("history_ins")),
    }


def _find_summary_files(run_dir: Path) -> List[Path]:
    summary_paths = list(run_dir.glob("**/summary.json"))
    summary_paths.extend(run_dir.glob("**/summarize.json"))
    # Keep deterministic order and remove duplicates.
    return sorted(set(summary_paths))


def _mean_curve_with_last_padding(curves: Sequence[Sequence[float]]) -> List[float]:
    valid = [list(c) for c in curves if c]
    if not valid:
        return []

    max_len = max(len(c) for c in valid)
    out: List[float] = []
    for idx in range(max_len):
        values: List[float] = []
        for curve in valid:
            if idx < len(curve):
                values.append(float(curve[idx]))
            else:
                values.append(float(curve[-1]))
        out.append(float(np.mean(values)))
    return out


def _curve_subtract(a: Sequence[float], b: Sequence[float]) -> List[float]:
    if not a and not b:
        return []
    merged = _mean_curve_with_last_padding([a, b])
    if not merged:
        return []

    max_len = max(len(a), len(b))
    out: List[float] = []
    for idx in range(max_len):
        va = float(a[idx] if idx < len(a) else a[-1]) if a else 0.0
        vb = float(b[idx] if idx < len(b) else b[-1]) if b else 0.0
        out.append(va - vb)
    return out


def _mean_scalar(values: Sequence[float]) -> Optional[float]:
    nums = [float(v) for v in values if not math.isnan(float(v)) and not math.isinf(float(v))]
    if not nums:
        return None
    return float(np.mean(nums))


def _mean_auc_over_samples(curves: Sequence[Sequence[float]]) -> Optional[float]:
    auc_values: List[float] = []
    for curve in curves:
        curve_list = [float(v) for v in curve]
        auc_value = _auc(curve_list)
        if auc_value is None:
            continue
        auc_values.append(float(auc_value))
    return _mean_scalar(auc_values)


def _metric_record(
    curve: Sequence[float],
    sample_curves: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    curve_list = [float(v) for v in curve]
    auc_value = _auc(curve_list) if curve_list else None
    mean_value = _mean_scalar(curve_list)
    auc_mean_over_samples = _mean_auc_over_samples(sample_curves or [])
    return {
        "num_points": len(curve_list),
        "mean": mean_value,
        "auc": auc_value,
        "auc_mean_over_samples": auc_mean_over_samples,
    }


def _imd_metric_record(
    ins_curve: Sequence[float],
    del_curve: Sequence[float],
    ins_sample_curves: Optional[Sequence[Sequence[float]]] = None,
    del_sample_curves: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    imd_curve = _curve_subtract(ins_curve, del_curve)
    metric = _metric_record(imd_curve)

    auc_ins_mean = _mean_auc_over_samples(ins_sample_curves or [])
    auc_del_mean = _mean_auc_over_samples(del_sample_curves or [])
    if auc_ins_mean is None or auc_del_mean is None:
        auc_imd_mean_samples = None
    else:
        auc_imd_mean_samples = float(auc_ins_mean - auc_del_mean)

    metric["auc_mean_over_samples"] = auc_imd_mean_samples
    return metric


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _safe_filename_token(name: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in name.strip())
    return token or "unknown"


def _fmt_num(value: Optional[float]) -> str:
    if value is None:
        return "nan"
    return f"{float(value):.4f}"


def _pick_metric_value(metric: object, preferred_keys: Sequence[str]) -> Optional[float]:
    if not isinstance(metric, dict):
        return None
    for key in preferred_keys:
        value = metric.get(key)
        if value is None:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(num) or math.isinf(num):
            continue
        return num
    return None


def _warm_palette(count: int) -> List[str]:
    if count <= 0:
        return []

    # Dark high-contrast palette for readability on white backgrounds.
    dark_base = [
        "#0B3C5D",
        "#8B1E3F",
        "#1F5D27",
        "#7A3E00",
        "#3D2C8D",
        "#005B5B",
        "#7F1D1D",
        "#2C3E50",
        "#5B2C6F",
        "#145A32",
    ]
    if count <= len(dark_base):
        return dark_base[:count]

    def _darken(hex_color: str, factor: float = 0.82) -> str:
        r, g, b = to_rgb(hex_color)
        return to_hex((r * factor, g * factor, b * factor))

    if count <= 20:
        cmap = plt.get_cmap("tab20", count)
        return [_darken(to_hex(cmap(i))) for i in range(count)]

    # For many approaches, spread hues uniformly.
    cmap = plt.get_cmap("hsv", count)
    return [_darken(to_hex(cmap(i))) for i in range(count)]


def _style_for_approach(label: str, default_color: str) -> Tuple[str, str, str, int]:
    lower = label.lower()
    if "both" in lower:
        return (default_color, "-", "o", 5)
    if "delete" in lower:
        return (default_color, "--", "s", 4)
    if "insert" in lower:
        return (default_color, "-.", "^", 4)
    if "clean" in lower:
        return ("#2e8b57", ":", "D", 3)
    return (default_color, "-", "o", 3)


def _apply_reference_axes_style(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, color="#cfd8dc", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=10)


def _markevery_indices(length: int, count: int = 15) -> Optional[List[int]]:
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


def _build_approach_color_map(approaches: Sequence[str]) -> Dict[str, str]:
    sorted_approaches = sorted(set(approaches))
    colors = _warm_palette(len(sorted_approaches))
    return {name: colors[idx] for idx, name in enumerate(sorted_approaches)}


def _plot_spearman_bar_comparison(
    scores_by_approach: Dict[str, float],
    approach_color_map: Dict[str, str],
    output_path: Path,
    title: str,
    dpi: int,
) -> Optional[str]:
    if not scores_by_approach:
        return None

    ordered = sorted(scores_by_approach.items(), key=lambda item: item[1])
    labels = [item[0] for item in ordered]
    values = [float(item[1]) for item in ordered]
    indices = np.arange(len(labels))
    colors = [approach_color_map.get(label, "#5B84B1") for label in labels]

    fig_w = max(9.0, len(labels) * 0.95 + 4.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 5.2))
    bars = ax.bar(indices, values, color=colors, alpha=0.95, edgecolor="#2c3e50", linewidth=0.7)

    min_idx = int(np.argmin(values))
    min_label = labels[min_idx]

    # Highlight the lowest-Spearman approach clearly.
    bars[min_idx].set_edgecolor("#111111")
    bars[min_idx].set_linewidth(2.0)
    bars[min_idx].set_hatch("///")

    y_min_data = float(min(values))
    y_max_data = float(max(values))
    span = max(1e-6, y_max_data - y_min_data)
    pad = max(0.015, span * 0.25)
    y_min = max(-1.0, y_min_data - pad)
    y_max = min(1.0, y_max_data + pad)
    if y_max <= y_min:
        y_min = max(-1.0, y_min_data - 0.05)
        y_max = min(1.0, y_max_data + 0.05)

    ax.set_ylim(y_min, y_max)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=26, ha="right")
    ax.set_ylabel("Mean Spearman")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)

    label_offset = max(0.01, (y_max - y_min) * 0.02)
    for idx, (rect, value) in enumerate(zip(bars, values)):
        ax.text(
            rect.get_x() + rect.get_width() * 0.5,
            rect.get_height() + label_offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if idx == min_idx else "normal",
            color="#111111",
        )

    ax.text(
        0.01,
        0.98,
        f"Lowest: {min_label} ({values[min_idx]:.3f})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#111111",
        bbox={"facecolor": "#f8f9fa", "edgecolor": "#111111", "boxstyle": "round,pad=0.25"},
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return min_label


def _plot_one_curve(
    curve: List[float],
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.2))

    if curve:
        x = np.linspace(0.0, 1.0, num=len(curve), endpoint=True)
        y = np.asarray(curve, dtype=float)
        ax.plot(x, y, linewidth=1.5, color="#1f77b4")
        ax.fill_between(x, y, 0.0, color="#1f77b4", alpha=0.28)
        auc_value = _auc(curve)
        if auc_value is not None:
            ax.text(
                0.5,
                0.5,
                f"AUC={auc_value:.4f}",
                color="#8B0000",
                fontsize=22,
                fontweight="bold",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_overlay_curves(
    clean_curve: List[float],
    adv_curve: List[float],
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.2))
    _apply_reference_axes_style(ax)

    if clean_curve:
        x_clean = np.linspace(0.0, 1.0, num=len(clean_curve), endpoint=True)
        y_clean = np.asarray(clean_curve, dtype=float)
        clean_markevery = _markevery_indices(len(clean_curve), 15)
        ax.plot(
            x_clean,
            y_clean,
            linewidth=2.2,
            color="#1f77b4",
            marker="o",
            markersize=5.2,
            markerfacecolor="white",
            markeredgecolor="#1f77b4",
            markeredgewidth=1.6,
            markevery=clean_markevery,
        )

    if adv_curve:
        x_adv = np.linspace(0.0, 1.0, num=len(adv_curve), endpoint=True)
        y_adv = np.asarray(adv_curve, dtype=float)
        adv_markevery = _markevery_indices(len(adv_curve), 15)
        ax.plot(
            x_adv,
            y_adv,
            linewidth=2.2,
            color="#d62728",
            marker="s",
            markersize=5.2,
            markerfacecolor="white",
            markeredgecolor="#d62728",
            markeredgewidth=1.6,
            markevery=adv_markevery,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_multi_curves(
    named_curves: Sequence[Tuple[str, List[float], str]],
    output_path: Path,
    dpi: int,
    y_label: str,
    y_min: float,
    y_max: float,
    marker_count: int,
    points_only: bool,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.0))
    _apply_reference_axes_style(ax)

    max_len = max((len(curve) for _, curve, _ in named_curves if curve), default=0)
    if max_len <= 0:
        plt.close(fig)
        return

    fixed_marker_count = 15

    for label, curve, color in named_curves:
        if not curve:
            continue
        markevery = _markevery_indices(len(curve), fixed_marker_count)
        if markevery is None:
            selected_idx = list(range(len(curve)))
        else:
            selected_idx = list(markevery)

        x_full = np.linspace(0.0, 1.0, num=len(curve), endpoint=True)
        y_full = np.asarray(curve, dtype=float)
        x = x_full[selected_idx]
        y = y_full[selected_idx]
        curve_color, curve_style, curve_marker, z_order = _style_for_approach(label, color)

        if points_only:
            ax.scatter(x, y, s=22, color=curve_color, label=label, zorder=z_order)
        else:
            ax.plot(
                x,
                y,
                linewidth=2.35,
                color=curve_color,
                label=label,
                linestyle=curve_style,
                marker=curve_marker,
                markersize=5.2,
                markerfacecolor="white",
                markeredgecolor=curve_color,
                markeredgewidth=1.5,
                zorder=z_order,
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_ylabel(y_label)
    ax.legend(loc="best", frameon=True, fancybox=True, framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_iteration_curves(
    named_curves: Sequence[Tuple[str, List[float], str]],
    output_path: Path,
    dpi: int,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.0))
    _apply_reference_axes_style(ax)

    max_len = max((len(curve) for _, curve, _ in named_curves if curve), default=0)
    if max_len <= 0:
        plt.close(fig)
        return

    y_all: List[float] = []
    for label, curve, color in named_curves:
        if not curve:
            continue
        x = np.arange(len(curve), dtype=float)
        y = np.asarray(curve, dtype=float)
        y_all.extend(float(v) for v in y)
        curve_color, curve_style, curve_marker, z_order = _style_for_approach(label, color)
        markevery = _markevery_indices(len(curve), 15)
        ax.plot(
            x,
            y,
            linewidth=2.25,
            color=curve_color,
            label=label,
            linestyle=curve_style,
            marker=curve_marker,
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=curve_color,
            markeredgewidth=1.3,
            markevery=markevery,
            zorder=z_order,
        )

    if not y_all:
        plt.close(fig)
        return

    y_min = float(min(y_all))
    y_max = float(max(y_all))
    if y_max <= y_min:
        pad = 0.05 if y_min == 0.0 else max(1e-3, abs(y_min) * 0.05)
        y_min -= pad
        y_max += pad
    else:
        span = y_max - y_min
        pad = max(1e-6, span * 0.05)
        y_min -= pad
        y_max += pad

    ax.set_xlim(0.0, float(max_len - 1))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(y_label)
    ax.legend(loc="best", frameon=True, fancybox=True, framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _split_summary_relpath(
    summary_path: Path,
    run_dir: Path,
    payload: Dict[str, object],
) -> Optional[Tuple[str, str, str]]:
    try:
        rel = summary_path.relative_to(run_dir)
    except ValueError:
        return None

    # Expected tail: .../class_name/image_name/summary.json
    if len(rel.parts) < 6:
        return None

    run_segments = list(rel.parts[:-3])
    if len(run_segments) < 3:
        return None

    model_name = payload.get("model") if isinstance(payload.get("model"), str) else None
    if model_name and model_name in run_segments:
        model_idx = run_segments.index(model_name)

        if model_idx == 0:
            # Legacy style: model/explain/approach[/...]
            explain_method = run_segments[1]
            approach_segments = run_segments[2:]
        else:
            # Current result style: explain/model/setting/approach[/...]
            explain_method = run_segments[model_idx - 1]
            approach_segments = run_segments[model_idx + 1 :]

        approach = "/".join(approach_segments) if approach_segments else "default"
        return model_name, explain_method, approach

    # Fallback when model field is missing or does not match path segments.
    model_name = run_segments[0]
    explain_method = run_segments[1]
    approach = "/".join(run_segments[2:]) if len(run_segments) > 2 else "default"
    return model_name, explain_method, approach


def _extract_model_name(
    summary_path: Path,
    run_dir: Path,
    payload: Dict[str, object],
) -> Optional[str]:
    location = _split_summary_relpath(summary_path, run_dir, payload)
    if location is not None:
        model_name, _, _ = location
        return model_name

    model_field = payload.get("model")
    if isinstance(model_field, str) and model_field.strip():
        return model_field.strip()
    return None


def _extract_explain_method(
    summary_path: Path,
    run_dir: Path,
    payload: Dict[str, object],
) -> Optional[str]:
    location = _split_summary_relpath(summary_path, run_dir, payload)
    if location is not None:
        _, explain_method, _ = location
        return explain_method

    for key in ("explain_method", "method", "explain"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_approach_folder_name(
    summary_path: Path,
    run_dir: Path,
    payload: Dict[str, object],
) -> Optional[str]:
    try:
        rel = summary_path.relative_to(run_dir)
    except ValueError:
        rel = None

    if rel is not None and len(rel.parts) >= 6:
        run_segments = list(rel.parts[:-3])
        model_name = payload.get("model") if isinstance(payload.get("model"), str) else None
        if model_name and model_name in run_segments:
            model_idx = run_segments.index(model_name)
            if model_idx == 0:
                # Legacy style: model/explain/approach[/...]
                if len(run_segments) > 2 and run_segments[2].strip():
                    return run_segments[2].strip()
            else:
                # Current style: explain/model/setting/approach[/...]
                if len(run_segments) > model_idx + 2 and run_segments[model_idx + 2].strip():
                    return run_segments[model_idx + 2].strip()

    location = _split_summary_relpath(summary_path, run_dir, payload)
    if location is not None:
        _, _, approach = location
        if approach:
            parts = [p.strip() for p in approach.split("/") if p.strip()]
            if len(parts) >= 2:
                return parts[1]
            if parts:
                return parts[0]

    for key in ("approach", "approach_name", "run_mode"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_method_output_dir(
    run_dir: Path,
    output_root: Optional[Path],
    model_name: str,
    explain_method: str,
) -> Path:
    base = output_root if output_root is not None else run_dir
    if (run_dir / explain_method / model_name).exists():
        return base / explain_method / model_name
    return base / model_name / explain_method


def _aggregate_and_plot(
    run_dir: Path,
    summary_paths: Sequence[Path],
    output_root: Optional[Path],
    dpi: int,
    allow_non_ok: bool,
    aggregate_y_min: float,
    aggregate_y_max: float,
    aggregate_marker_count: int,
    aggregate_points_only: bool,
    run_model: Optional[str],
    run_explain: Optional[str],
    run_approach: Optional[str],
) -> None:
    per_run_curves: Dict[Tuple[str, str, str], Dict[str, List[List[float]]]] = {}
    per_run_spearman: Dict[Tuple[str, str, str], List[List[float]]] = {}
    per_run_objectives: Dict[Tuple[str, str, str], Dict[str, List[List[float]]]] = {}
    clean_reference: Dict[Tuple[str, str], Dict[str, List[List[float]]]] = {}
    method_txt_rows: List[str] = []
    detailed_txt_rows: List[str] = []
    approach_metric_pool: Dict[str, Dict[str, List[float]]] = {}
    full_spearman_pool: Dict[str, List[float]] = {}
    model_spearman_pool: Dict[str, Dict[str, List[float]]] = {}
    full_adv_del_curves: List[List[float]] = []
    full_adv_ins_curves: List[List[float]] = []

    for summary_path in summary_paths:
        payload = _read_json(summary_path)
        if payload is None:
            continue
        if not allow_non_ok and payload.get("status") not in {None, "ok"}:
            continue

        if run_model:
            model_name_in_payload = _extract_model_name(summary_path, run_dir, payload)
            if model_name_in_payload != run_model:
                continue

        if run_explain:
            explain_in_payload = _extract_explain_method(summary_path, run_dir, payload)
            if explain_in_payload != run_explain:
                continue

        if run_approach:
            approach_in_payload = _extract_approach_folder_name(summary_path, run_dir, payload)
            if approach_in_payload != run_approach:
                continue

        location = _split_summary_relpath(summary_path, run_dir, payload)
        if location is None:
            continue
        model_name, explain_method, approach = location
        run_key = (model_name, explain_method, approach)
        method_key = (model_name, explain_method)

        curves = _extract_curves(payload)
        if curves is not None:
            holder = per_run_curves.setdefault(
                run_key,
                {
                    "adv_del": [],
                    "adv_ins": [],
                    "clean_del": [],
                    "clean_ins": [],
                },
            )
            for key in ("adv_del", "adv_ins", "clean_del", "clean_ins"):
                if curves[key]:
                    holder[key].append(curves[key])

            clean_holder = clean_reference.setdefault(
                method_key,
                {
                    "clean_del": [],
                    "clean_ins": [],
                },
            )
            if curves["clean_del"]:
                clean_holder["clean_del"].append(curves["clean_del"])
            if curves["clean_ins"]:
                clean_holder["clean_ins"].append(curves["clean_ins"])
            if curves["adv_del"]:
                full_adv_del_curves.append(curves["adv_del"])
            if curves["adv_ins"]:
                full_adv_ins_curves.append(curves["adv_ins"])

        sp = _extract_spearman(payload)
        if sp:
            per_run_spearman.setdefault(run_key, []).append(sp)

        spearman_scalar = _extract_spearman_scalar(payload)
        if spearman_scalar is not None:
            full_spearman_pool.setdefault(approach, []).append(float(spearman_scalar))
            model_bucket = model_spearman_pool.setdefault(model_name, {})
            model_bucket.setdefault(approach, []).append(float(spearman_scalar))

        objective_histories = _extract_objective_histories(payload)
        obj_holder = per_run_objectives.setdefault(
            run_key,
            {
                "margin": [],
                "saliency": [],
                "deletion_obj": [],
                "insertion_obj": [],
            },
        )
        for obj_name, obj_curve in objective_histories.items():
            if obj_curve:
                obj_holder[obj_name].append(obj_curve)

    # Build per-run means first: sample-average inside each (model, method, approach).
    per_run_mean: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}
    for run_key, holder in per_run_curves.items():
        per_run_mean[run_key] = {
            "adv_del": _mean_curve_with_last_padding(holder["adv_del"]),
            "adv_ins": _mean_curve_with_last_padding(holder["adv_ins"]),
            "clean_del": _mean_curve_with_last_padding(holder["clean_del"]),
            "clean_ins": _mean_curve_with_last_padding(holder["clean_ins"]),
        }

    per_run_spearman_mean: Dict[Tuple[str, str, str], List[float]] = {}
    for run_key, curves in per_run_spearman.items():
        per_run_spearman_mean[run_key] = _mean_curve_with_last_padding(curves)

    per_run_objective_mean: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}
    for run_key, objectives in per_run_objectives.items():
        per_run_objective_mean[run_key] = {
            "margin": _mean_curve_with_last_padding(objectives["margin"]),
            "saliency": _mean_curve_with_last_padding(objectives["saliency"]),
            "deletion_obj": _mean_curve_with_last_padding(objectives["deletion_obj"]),
            "insertion_obj": _mean_curve_with_last_padding(objectives["insertion_obj"]),
        }

    # Plot per (model, method): compare approaches + one clean reference.
    model_method_keys = sorted({(m, e) for (m, e, _) in per_run_mean.keys()} | {(m, e) for (m, e, _) in per_run_spearman_mean.keys()})
    for model_name, explain_method in model_method_keys:
        approaches = sorted({a for (m, e, a) in per_run_mean.keys() if m == model_name and e == explain_method} | {a for (m, e, a) in per_run_spearman_mean.keys() if m == model_name and e == explain_method})
        if not approaches:
            continue

        warm_colors = _warm_palette(len(approaches))
        color_by_approach = {name: warm_colors[idx] for idx, name in enumerate(approaches)}

        method_dir = _resolve_method_output_dir(
            run_dir=run_dir,
            output_root=output_root,
            model_name=model_name,
            explain_method=explain_method,
        )

        clean_holder = clean_reference.get((model_name, explain_method), {"clean_del": [], "clean_ins": []})
        mean_clean_del = _mean_curve_with_last_padding(clean_holder.get("clean_del", []))
        mean_clean_ins = _mean_curve_with_last_padding(clean_holder.get("clean_ins", []))

        del_curves: List[Tuple[str, List[float], str]] = []
        ins_curves: List[Tuple[str, List[float], str]] = []

        if mean_clean_del:
            del_curves.append(("clean", mean_clean_del, "#2E8B57"))
        if mean_clean_ins:
            ins_curves.append(("clean", mean_clean_ins, "#2E8B57"))

        for approach in approaches:
            run_key = (model_name, explain_method, approach)
            mean_block = per_run_mean.get(run_key)
            if mean_block is None:
                continue
            if mean_block["adv_del"]:
                del_curves.append((approach, mean_block["adv_del"], color_by_approach[approach]))
            if mean_block["adv_ins"]:
                ins_curves.append((approach, mean_block["adv_ins"], color_by_approach[approach]))

        _plot_multi_curves(
            del_curves,
            method_dir / "avg_compare_approaches_deletion.png",
            dpi=dpi,
            y_label="Deletion score",
            y_min=aggregate_y_min,
            y_max=aggregate_y_max,
            marker_count=aggregate_marker_count,
            points_only=aggregate_points_only,
        )
        _plot_multi_curves(
            ins_curves,
            method_dir / "avg_compare_approaches_insertion.png",
            dpi=dpi,
            y_label="Insertion score",
            y_min=aggregate_y_min,
            y_max=aggregate_y_max,
            marker_count=aggregate_marker_count,
            points_only=aggregate_points_only,
        )

        spearman_curves: List[Tuple[str, List[float], str]] = []
        for approach in approaches:
            run_key = (model_name, explain_method, approach)
            mean_sp = per_run_spearman_mean.get(run_key, [])
            if mean_sp:
                spearman_curves.append((approach, mean_sp, color_by_approach[approach]))

        _plot_multi_curves(
            spearman_curves,
            method_dir / "avg_compare_approaches_spearman.png",
            dpi=dpi,
            y_label="Spearman",
            y_min=aggregate_y_min,
            y_max=aggregate_y_max,
            marker_count=aggregate_marker_count,
            points_only=aggregate_points_only,
        )

        objective_del_curves: List[Tuple[str, List[float], str]] = []
        objective_ins_curves: List[Tuple[str, List[float], str]] = []
        for approach in approaches:
            run_key = (model_name, explain_method, approach)
            objective_block = per_run_objective_mean.get(run_key, {})
            del_obj_curve = list(objective_block.get("deletion_obj", []))
            ins_obj_curve = list(objective_block.get("insertion_obj", []))
            if del_obj_curve:
                objective_del_curves.append((approach, del_obj_curve, color_by_approach[approach]))
            if ins_obj_curve:
                objective_ins_curves.append((approach, ins_obj_curve, color_by_approach[approach]))

        _plot_iteration_curves(
            objective_del_curves,
            method_dir / "avg_compare_approaches_objective_del_by_iteration.png",
            dpi=dpi,
            y_label="Loss del",
        )
        _plot_iteration_curves(
            objective_ins_curves,
            method_dir / "avg_compare_approaches_objective_ins_by_iteration.png",
            dpi=dpi,
            y_label="Loss ins",
        )

        model_method_metrics: Dict[str, Any] = {
            "model": model_name,
            "explain_method": explain_method,
            "clean": {
                "deletion": _metric_record(
                    mean_clean_del,
                    sample_curves=clean_holder.get("clean_del", []),
                ),
                "insertion": _metric_record(
                    mean_clean_ins,
                    sample_curves=clean_holder.get("clean_ins", []),
                ),
                "imd_ins_minus_del": _imd_metric_record(
                    mean_clean_ins,
                    mean_clean_del,
                    ins_sample_curves=clean_holder.get("clean_ins", []),
                    del_sample_curves=clean_holder.get("clean_del", []),
                ),
            },
            "approaches": {},
        }
        for approach in approaches:
            run_key = (model_name, explain_method, approach)
            mean_block = per_run_mean.get(run_key, {})
            raw_block = per_run_curves.get(run_key, {})
            del_curve = list(mean_block.get("adv_del", []))
            ins_curve = list(mean_block.get("adv_ins", []))
            spearman_curve = list(per_run_spearman_mean.get(run_key, []))
            raw_spearman = per_run_spearman.get(run_key, [])

            objective_block = per_run_objective_mean.get(run_key, {})
            raw_objective_block = per_run_objectives.get(run_key, {})
            model_method_metrics["approaches"][approach] = {
                "faithfulness": {
                    "deletion": _metric_record(del_curve, sample_curves=raw_block.get("adv_del", [])),
                    "insertion": _metric_record(ins_curve, sample_curves=raw_block.get("adv_ins", [])),
                    "imd_ins_minus_del": _imd_metric_record(
                        ins_curve,
                        del_curve,
                        ins_sample_curves=raw_block.get("adv_ins", []),
                        del_sample_curves=raw_block.get("adv_del", []),
                    ),
                },
                "spearman": _metric_record(spearman_curve, sample_curves=raw_spearman),
                "objectives": {
                    "margin": _metric_record(
                        objective_block.get("margin", []),
                        sample_curves=raw_objective_block.get("margin", []),
                    ),
                    "saliency": _metric_record(
                        objective_block.get("saliency", []),
                        sample_curves=raw_objective_block.get("saliency", []),
                    ),
                    "deletion_obj": _metric_record(
                        objective_block.get("deletion_obj", []),
                        sample_curves=raw_objective_block.get("deletion_obj", []),
                    ),
                    "insertion_obj": _metric_record(
                        objective_block.get("insertion_obj", []),
                        sample_curves=raw_objective_block.get("insertion_obj", []),
                    ),
                },
            }

        clean_metric_block = model_method_metrics.get("clean", {})
        clean_ins_metric = clean_metric_block.get("insertion", {}) if isinstance(clean_metric_block, dict) else {}
        clean_del_metric = clean_metric_block.get("deletion", {}) if isinstance(clean_metric_block, dict) else {}
        clean_imd_metric = clean_metric_block.get("imd_ins_minus_del", {}) if isinstance(clean_metric_block, dict) else {}

        clean_ins_auc = clean_ins_metric.get("auc_mean_over_samples") if isinstance(clean_ins_metric, dict) else None
        clean_del_auc = clean_del_metric.get("auc_mean_over_samples") if isinstance(clean_del_metric, dict) else None
        clean_imd_auc = clean_imd_metric.get("auc_mean_over_samples") if isinstance(clean_imd_metric, dict) else None

        if clean_ins_auc is None and isinstance(clean_ins_metric, dict):
            clean_ins_auc = clean_ins_metric.get("auc")
        if clean_del_auc is None and isinstance(clean_del_metric, dict):
            clean_del_auc = clean_del_metric.get("auc")
        if clean_imd_auc is None and isinstance(clean_imd_metric, dict):
            clean_imd_auc = clean_imd_metric.get("auc")

        detailed_txt_rows.append(
            f"{explain_method} & {model_name} & clean & {_fmt_num(clean_del_auc)} & {_fmt_num(clean_ins_auc)} & {_fmt_num(clean_imd_auc)}"
        )

        for approach_name, approach_block in model_method_metrics.get("approaches", {}).items():
            if not isinstance(approach_block, dict):
                continue
            faithfulness_block = approach_block.get("faithfulness", {})
            if not isinstance(faithfulness_block, dict):
                continue

            ins_metric = faithfulness_block.get("insertion", {})
            del_metric = faithfulness_block.get("deletion", {})
            imd_metric = faithfulness_block.get("imd_ins_minus_del", {})

            ins_auc = ins_metric.get("auc_mean_over_samples") if isinstance(ins_metric, dict) else None
            del_auc = del_metric.get("auc_mean_over_samples") if isinstance(del_metric, dict) else None
            imd_auc = imd_metric.get("auc_mean_over_samples") if isinstance(imd_metric, dict) else None

            if ins_auc is None and isinstance(ins_metric, dict):
                ins_auc = ins_metric.get("auc")
            if del_auc is None and isinstance(del_metric, dict):
                del_auc = del_metric.get("auc")
            if imd_auc is None and isinstance(imd_metric, dict):
                imd_auc = imd_metric.get("auc")

            detailed_txt_rows.append(
                f"{explain_method} & {model_name} & {approach_name} & {_fmt_num(del_auc)} & {_fmt_num(ins_auc)} & {_fmt_num(imd_auc)}"
            )

        _write_json(method_dir / "avg_compare_approaches_metrics.json", model_method_metrics)
        model_token = _safe_filename_token(model_name)
        _write_json(
            method_dir / f"avg_compare_approaches_metrics_{model_token}.json",
            model_method_metrics,
        )

    # Plot per explain method across models: average over model-level means.
    methods = sorted({e for (_, e, _) in per_run_mean.keys()} | {e for (_, e, _) in per_run_spearman_mean.keys()})
    for explain_method in methods:
        approaches = sorted({a for (_, e, a) in per_run_mean.keys() if e == explain_method} | {a for (_, e, a) in per_run_spearman_mean.keys() if e == explain_method})
        if not approaches:
            continue

        warm_colors = _warm_palette(len(approaches))
        color_by_approach = {name: warm_colors[idx] for idx, name in enumerate(approaches)}

        # Collect curves from each model for the same approach, then average.
        method_adv_del: Dict[str, List[List[float]]] = {a: [] for a in approaches}
        method_adv_ins: Dict[str, List[List[float]]] = {a: [] for a in approaches}
        method_spearman: Dict[str, List[List[float]]] = {a: [] for a in approaches}
        method_clean_by_model: Dict[str, Tuple[List[float], List[float]]] = {}

        for (model_name, method_name, approach), curves in per_run_mean.items():
            if method_name != explain_method:
                continue
            if curves["adv_del"]:
                method_adv_del[approach].append(curves["adv_del"])
            if curves["adv_ins"]:
                method_adv_ins[approach].append(curves["adv_ins"])

            if model_name not in method_clean_by_model:
                clean_block = clean_reference.get((model_name, method_name), {"clean_del": [], "clean_ins": []})
                cdel = _mean_curve_with_last_padding(clean_block.get("clean_del", []))
                cins = _mean_curve_with_last_padding(clean_block.get("clean_ins", []))
                method_clean_by_model[model_name] = (cdel, cins)

        for (model_name, method_name, approach), sp_curve in per_run_spearman_mean.items():
            if method_name != explain_method or not sp_curve:
                continue
            method_spearman[approach].append(sp_curve)

        method_dir = run_dir / "_method_aggregates" / explain_method
        if output_root is not None:
            method_dir = output_root / "_method_aggregates" / explain_method

        method_clean_del = [pair[0] for pair in method_clean_by_model.values() if pair[0]]
        method_clean_ins = [pair[1] for pair in method_clean_by_model.values() if pair[1]]
        mean_clean_del = _mean_curve_with_last_padding(method_clean_del)
        mean_clean_ins = _mean_curve_with_last_padding(method_clean_ins)

        del_curves: List[Tuple[str, List[float], str]] = []
        ins_curves: List[Tuple[str, List[float], str]] = []
        if mean_clean_del:
            del_curves.append(("clean", mean_clean_del, "#2E8B57"))
        if mean_clean_ins:
            ins_curves.append(("clean", mean_clean_ins, "#2E8B57"))

        for approach in approaches:
            mean_del = _mean_curve_with_last_padding(method_adv_del.get(approach, []))
            mean_ins = _mean_curve_with_last_padding(method_adv_ins.get(approach, []))
            if mean_del:
                del_curves.append((approach, mean_del, color_by_approach[approach]))
            if mean_ins:
                ins_curves.append((approach, mean_ins, color_by_approach[approach]))

        _plot_multi_curves(
            del_curves,
            method_dir / "avg_models_compare_approaches_deletion.png",
            dpi=dpi,
            y_label="Deletion score",
            y_min=aggregate_y_min,
            y_max=aggregate_y_max,
            marker_count=aggregate_marker_count,
            points_only=aggregate_points_only,
        )
        _plot_multi_curves(
            ins_curves,
            method_dir / "avg_models_compare_approaches_insertion.png",
            dpi=dpi,
            y_label="Insertion score",
            y_min=aggregate_y_min,
            y_max=aggregate_y_max,
            marker_count=aggregate_marker_count,
            points_only=aggregate_points_only,
        )

        spearman_curves: List[Tuple[str, List[float], str]] = []
        for approach in approaches:
            mean_sp = _mean_curve_with_last_padding(method_spearman.get(approach, []))
            if mean_sp:
                spearman_curves.append((approach, mean_sp, color_by_approach[approach]))

        _plot_multi_curves(
            spearman_curves,
            method_dir / "avg_models_compare_approaches_spearman.png",
            dpi=dpi,
            y_label="Spearman",
            y_min=aggregate_y_min,
            y_max=aggregate_y_max,
            marker_count=aggregate_marker_count,
            points_only=aggregate_points_only,
        )

        method_objective_del_curves: List[Tuple[str, List[float], str]] = []
        method_objective_ins_curves: List[Tuple[str, List[float], str]] = []
        for approach in approaches:
            objective_del_curves: List[List[float]] = []
            objective_ins_curves: List[List[float]] = []
            for (m_name, m_method, m_approach), obj_block in per_run_objective_mean.items():
                if m_method != explain_method or m_approach != approach:
                    continue
                if obj_block.get("deletion_obj"):
                    objective_del_curves.append(obj_block["deletion_obj"])
                if obj_block.get("insertion_obj"):
                    objective_ins_curves.append(obj_block["insertion_obj"])

            mean_objective_del = _mean_curve_with_last_padding(objective_del_curves)
            mean_objective_ins = _mean_curve_with_last_padding(objective_ins_curves)
            if mean_objective_del:
                method_objective_del_curves.append((approach, mean_objective_del, color_by_approach[approach]))
            if mean_objective_ins:
                method_objective_ins_curves.append((approach, mean_objective_ins, color_by_approach[approach]))

        _plot_iteration_curves(
            method_objective_del_curves,
            method_dir / "avg_models_compare_approaches_objective_del_by_iteration.png",
            dpi=dpi,
            y_label="Loss del",
        )
        _plot_iteration_curves(
            method_objective_ins_curves,
            method_dir / "avg_models_compare_approaches_objective_ins_by_iteration.png",
            dpi=dpi,
            y_label="Loss ins",
        )

        method_metrics: Dict[str, Any] = {
            "explain_method": explain_method,
            "clean": {
                "deletion": _metric_record(mean_clean_del, sample_curves=method_clean_del),
                "insertion": _metric_record(mean_clean_ins, sample_curves=method_clean_ins),
                "imd_ins_minus_del": _imd_metric_record(
                    mean_clean_ins,
                    mean_clean_del,
                    ins_sample_curves=method_clean_ins,
                    del_sample_curves=method_clean_del,
                ),
            },
            "approaches": {},
        }

        for approach in approaches:
            mean_del = _mean_curve_with_last_padding(method_adv_del.get(approach, []))
            mean_ins = _mean_curve_with_last_padding(method_adv_ins.get(approach, []))
            mean_sp = _mean_curve_with_last_padding(method_spearman.get(approach, []))

            objective_margin_curves: List[List[float]] = []
            objective_saliency_curves: List[List[float]] = []
            objective_del_curves: List[List[float]] = []
            objective_ins_curves: List[List[float]] = []
            for (m_name, m_method, m_approach), obj_block in per_run_objective_mean.items():
                if m_method != explain_method or m_approach != approach:
                    continue
                if obj_block.get("margin"):
                    objective_margin_curves.append(obj_block["margin"])
                if obj_block.get("saliency"):
                    objective_saliency_curves.append(obj_block["saliency"])
                if obj_block.get("deletion_obj"):
                    objective_del_curves.append(obj_block["deletion_obj"])
                if obj_block.get("insertion_obj"):
                    objective_ins_curves.append(obj_block["insertion_obj"])

            method_metrics["approaches"][approach] = {
                "faithfulness": {
                    "deletion": _metric_record(mean_del, sample_curves=method_adv_del.get(approach, [])),
                    "insertion": _metric_record(mean_ins, sample_curves=method_adv_ins.get(approach, [])),
                    "imd_ins_minus_del": _imd_metric_record(
                        mean_ins,
                        mean_del,
                        ins_sample_curves=method_adv_ins.get(approach, []),
                        del_sample_curves=method_adv_del.get(approach, []),
                    ),
                },
                "spearman": _metric_record(mean_sp, sample_curves=method_spearman.get(approach, [])),
                "objectives": {
                    "margin": _metric_record(
                        _mean_curve_with_last_padding(objective_margin_curves),
                        sample_curves=objective_margin_curves,
                    ),
                    "saliency": _metric_record(
                        _mean_curve_with_last_padding(objective_saliency_curves),
                        sample_curves=objective_saliency_curves,
                    ),
                    "deletion_obj": _metric_record(
                        _mean_curve_with_last_padding(objective_del_curves),
                        sample_curves=objective_del_curves,
                    ),
                    "insertion_obj": _metric_record(
                        _mean_curve_with_last_padding(objective_ins_curves),
                        sample_curves=objective_ins_curves,
                    ),
                },
            }

        _write_json(method_dir / "avg_models_compare_approaches_metrics.json", method_metrics)

        clean_block = method_metrics.get("clean", {})
        ins_metric = clean_block.get("insertion", {}) if isinstance(clean_block, dict) else {}
        del_metric = clean_block.get("deletion", {}) if isinstance(clean_block, dict) else {}
        imd_metric = clean_block.get("imd_ins_minus_del", {}) if isinstance(clean_block, dict) else {}

        ins_auc = ins_metric.get("auc_mean_over_samples") if isinstance(ins_metric, dict) else None
        del_auc = del_metric.get("auc_mean_over_samples") if isinstance(del_metric, dict) else None
        imd_auc = imd_metric.get("auc_mean_over_samples") if isinstance(imd_metric, dict) else None

        if ins_auc is None and isinstance(ins_metric, dict):
            ins_auc = ins_metric.get("auc")
        if del_auc is None and isinstance(del_metric, dict):
            del_auc = del_metric.get("auc")
        if imd_auc is None and isinstance(imd_metric, dict):
            imd_auc = imd_metric.get("auc")

        method_txt_rows.append(
            f"{explain_method} & {_fmt_num(del_auc)} & {_fmt_num(ins_auc)} & {_fmt_num(imd_auc)}"
        )

        for approach_name, approach_block in method_metrics.get("approaches", {}).items():
            if not isinstance(approach_block, dict):
                continue

            faithfulness_block = approach_block.get("faithfulness", {})
            spearman_metric = approach_block.get("spearman", {})
            if not isinstance(faithfulness_block, dict):
                continue

            del_metric = faithfulness_block.get("deletion", {})
            ins_metric = faithfulness_block.get("insertion", {})
            imd_metric = faithfulness_block.get("imd_ins_minus_del", {})

            del_value = _pick_metric_value(del_metric, ("auc_mean_over_samples", "auc", "mean"))
            ins_value = _pick_metric_value(ins_metric, ("auc_mean_over_samples", "auc", "mean"))
            imd_value = _pick_metric_value(imd_metric, ("auc_mean_over_samples", "auc", "mean"))
            spearman_value = _pick_metric_value(spearman_metric, ("mean", "auc_mean_over_samples", "auc"))

            pool = approach_metric_pool.setdefault(
                approach_name,
                {"del": [], "ins": [], "imd": [], "spearman": []},
            )
            if del_value is not None:
                pool["del"].append(float(del_value))
            if ins_value is not None:
                pool["ins"].append(float(ins_value))
            if imd_value is not None:
                pool["imd"].append(float(imd_value))
            if spearman_value is not None:
                pool["spearman"].append(float(spearman_value))

    txt_root = output_root if output_root is not None else run_dir
    txt_path = txt_root / "_method_aggregates" / "summary_ins_del_imd.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("explain_method & del & ins & imd\n")
        for row in sorted(method_txt_rows):
            f.write(row + "\n")

    detailed_txt_path = txt_root / "_method_aggregates" / "summary_ins_del_imd_by_model_approach.txt"
    with open(detailed_txt_path, "w", encoding="utf-8") as f:
        f.write("explain_method & model_name & approach & del & ins & imd\n")
        for row in sorted(detailed_txt_rows):
            f.write(row + "\n")

    approach_summary_rows: List[str] = []
    approach_summary_payload: Dict[str, Dict[str, Optional[float]]] = {}
    for approach_name in sorted(approach_metric_pool.keys()):
        pool = approach_metric_pool[approach_name]
        del_mean = _mean_scalar(pool.get("del", []))
        ins_mean = _mean_scalar(pool.get("ins", []))
        imd_mean = _mean_scalar(pool.get("imd", []))
        spearman_mean = _mean_scalar(pool.get("spearman", []))

        approach_summary_rows.append(
            f"{approach_name} & {_fmt_num(del_mean)} & {_fmt_num(ins_mean)} & {_fmt_num(imd_mean)} & {_fmt_num(spearman_mean)}"
        )
        approach_summary_payload[approach_name] = {
            "del": del_mean,
            "ins": ins_mean,
            "imd": imd_mean,
            "spearman": spearman_mean,
            "num_entries": len(pool.get("spearman", [])),
        }

    approach_txt_path = txt_root / "_method_aggregates" / "summary_ins_del_imd_spearman_by_approach.txt"
    with open(approach_txt_path, "w", encoding="utf-8") as f:
        f.write("approach & del & ins & imd & spearman\n")
        for row in approach_summary_rows:
            f.write(row + "\n")

    _write_json(
        txt_root / "_method_aggregates" / "summary_ins_del_imd_spearman_by_approach.json",
        {"approaches": approach_summary_payload},
    )

    # Full average curves across all filtered samples/runs:
    # insertion/deletion stay grouped by approach, and include clean references.
    full_adv_del_by_approach: Dict[str, List[List[float]]] = {}
    full_adv_ins_by_approach: Dict[str, List[List[float]]] = {}
    for (model_name, explain_method, approach), mean_block in per_run_mean.items():
        del_curve = list(mean_block.get("adv_del", []))
        ins_curve = list(mean_block.get("adv_ins", []))
        if del_curve:
            full_adv_del_by_approach.setdefault(approach, []).append(del_curve)
        if ins_curve:
            full_adv_ins_by_approach.setdefault(approach, []).append(ins_curve)

    full_clean_del_sources: List[List[float]] = []
    full_clean_ins_sources: List[List[float]] = []
    for clean_block in clean_reference.values():
        cdel = _mean_curve_with_last_padding(clean_block.get("clean_del", []))
        cins = _mean_curve_with_last_padding(clean_block.get("clean_ins", []))
        if cdel:
            full_clean_del_sources.append(cdel)
        if cins:
            full_clean_ins_sources.append(cins)

    full_mean_clean_del = _mean_curve_with_last_padding(full_clean_del_sources)
    full_mean_clean_ins = _mean_curve_with_last_padding(full_clean_ins_sources)

    full_approaches = sorted(set(full_adv_del_by_approach.keys()) | set(full_adv_ins_by_approach.keys()))
    full_color_map = _build_approach_color_map(full_approaches)

    full_del_curves: List[Tuple[str, List[float], str]] = []
    full_ins_curves: List[Tuple[str, List[float], str]] = []

    if full_mean_clean_del:
        full_del_curves.append(("clean", full_mean_clean_del, "#2E8B57"))
    if full_mean_clean_ins:
        full_ins_curves.append(("clean", full_mean_clean_ins, "#2E8B57"))

    full_curve_payload: Dict[str, Any] = {
        "clean": {
            "deletion": full_mean_clean_del,
            "insertion": full_mean_clean_ins,
        },
        "approaches": {},
    }
    for approach in full_approaches:
        mean_del = _mean_curve_with_last_padding(full_adv_del_by_approach.get(approach, []))
        mean_ins = _mean_curve_with_last_padding(full_adv_ins_by_approach.get(approach, []))
        if mean_del:
            full_del_curves.append((approach, mean_del, full_color_map[approach]))
        if mean_ins:
            full_ins_curves.append((approach, mean_ins, full_color_map[approach]))

        full_curve_payload["approaches"][approach] = {
            "deletion": mean_del,
            "insertion": mean_ins,
            "num_del_curves": len(full_adv_del_by_approach.get(approach, [])),
            "num_ins_curves": len(full_adv_ins_by_approach.get(approach, [])),
            "color": full_color_map.get(approach),
        }

    _plot_multi_curves(
        full_del_curves,
        txt_root / "_method_aggregates" / "full_avg_compare_approaches_deletion.png",
        dpi=dpi,
        y_label="Deletion score",
        y_min=aggregate_y_min,
        y_max=aggregate_y_max,
        marker_count=aggregate_marker_count,
        points_only=aggregate_points_only,
    )
    _plot_multi_curves(
        full_ins_curves,
        txt_root / "_method_aggregates" / "full_avg_compare_approaches_insertion.png",
        dpi=dpi,
        y_label="Insertion score",
        y_min=aggregate_y_min,
        y_max=aggregate_y_max,
        marker_count=aggregate_marker_count,
        points_only=aggregate_points_only,
    )

    _write_json(
        txt_root / "_method_aggregates" / "full_avg_compare_approaches_curves.json",
        full_curve_payload,
    )

    # Spearman comparison bars across approaches: full + per-model.
    full_spearman_mean = {
        approach: mean_value
        for approach, mean_value in ((k, _mean_scalar(v)) for k, v in full_spearman_pool.items())
        if mean_value is not None
    }
    per_model_spearman_mean: Dict[str, Dict[str, float]] = {}
    for model_name, bucket in model_spearman_pool.items():
        mean_block = {
            approach: mean_value
            for approach, mean_value in ((k, _mean_scalar(v)) for k, v in bucket.items())
            if mean_value is not None
        }
        if mean_block:
            per_model_spearman_mean[model_name] = mean_block

    all_approaches = set(full_spearman_mean.keys())
    for bucket in per_model_spearman_mean.values():
        all_approaches.update(bucket.keys())
    approach_color_map = _build_approach_color_map(sorted(all_approaches))

    spearman_chart_dir = txt_root / "_method_aggregates" / "spearman_comparison"
    spearman_chart_dir.mkdir(parents=True, exist_ok=True)

    lowest_tracker: Dict[str, Optional[str]] = {}
    lowest_tracker["full"] = _plot_spearman_bar_comparison(
        full_spearman_mean,
        approach_color_map,
        spearman_chart_dir / "spearman_compare_approaches_full.png",
        title="Spearman by Approach (All Models)",
        dpi=dpi,
    )

    for model_name in sorted(per_model_spearman_mean.keys()):
        model_token = _safe_filename_token(model_name)
        lowest_tracker[model_name] = _plot_spearman_bar_comparison(
            per_model_spearman_mean[model_name],
            approach_color_map,
            spearman_chart_dir / f"spearman_compare_approaches_{model_token}.png",
            title=f"Spearman by Approach ({model_name})",
            dpi=dpi,
        )

    _write_json(
        spearman_chart_dir / "spearman_compare_approaches_summary.json",
        {
            "full": full_spearman_mean,
            "per_model": per_model_spearman_mean,
            "lowest_spearman_approach": lowest_tracker,
            "approach_colors": approach_color_map,
        },
    )


def _iter_targets(summary_paths: Iterable[Path]) -> Iterable[Path]:
    for path in summary_paths:
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-sample deletion/insertion curve images from summary files. "
            "Save separate images for deletion/insertion and clean/adv curves, without title/legend."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run folder path. Script recursively scans summary.json/summarize.json.",
    )
    parser.add_argument(
        "--run-model",
        type=str,
        default=None,
        help="Optional model name filter (exact match). If set, only this model is processed.",
    )
    parser.add_argument(
        "--run-explain",
        type=str,
        default=None,
        help="Optional explain method filter (exact match). If set, only this explain method is processed.",
    )
    parser.add_argument(
        "--run-approach",
        type=str,
        default=None,
        help="Optional approach folder filter (exact match), e.g. insert_oriented.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional root to write outputs. Default: save inside each sample folder.",
    )
    parser.add_argument(
        "--deletion-clean-name",
        type=str,
        default="deletion_clean.png",
        help="Output filename for clean deletion curve image.",
    )
    parser.add_argument(
        "--deletion-adv-name",
        type=str,
        default="deletion_adv.png",
        help="Output filename for adv deletion curve image.",
    )
    parser.add_argument(
        "--insertion-clean-name",
        type=str,
        default="insertion_clean.png",
        help="Output filename for clean insertion curve image.",
    )
    parser.add_argument(
        "--insertion-adv-name",
        type=str,
        default="insertion_adv.png",
        help="Output filename for adv insertion curve image.",
    )
    parser.add_argument(
        "--deletion-overlay-name",
        type=str,
        default="deletion_overlay.png",
        help="Output filename for overlay deletion curve image (clean blue, adv red).",
    )
    parser.add_argument(
        "--insertion-overlay-name",
        type=str,
        default="insertion_overlay.png",
        help="Output filename for overlay insertion curve image (clean blue, adv red).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--allow-non-ok",
        action="store_true",
        help="Also process summaries whose status != 'ok'.",
    )
    parser.add_argument(
        "--save-sample-plots",
        action="store_true",
        help="Save per-sample plots. Default is off for faster aggregate-only runs.",
    )
    parser.add_argument(
        "--skip-sample-plots",
        action="store_true",
        help="Deprecated compatibility flag. If provided, disables per-sample plots.",
    )
    parser.add_argument(
        "--aggregate-y-min",
        type=float,
        default=-0.02,
        help="Y-axis lower bound for aggregate comparison plots (zoom control).",
    )
    parser.add_argument(
        "--aggregate-y-max",
        type=float,
        default=1.02,
        help="Y-axis upper bound for aggregate comparison plots (zoom control).",
    )
    parser.add_argument(
        "--aggregate-marker-count",
        type=int,
        default=100,
        help="Number of sampled points per aggregate average curve.",
    )
    parser.add_argument(
        "--aggregate-points-only",
        action="store_true",
        help="Plot sampled points only (no connecting line) for aggregate curves.",
    )
    args = parser.parse_args()

    if args.aggregate_y_min >= args.aggregate_y_max:
        raise ValueError("aggregate-y-min must be smaller than aggregate-y-max")
    if args.aggregate_marker_count < 0:
        raise ValueError("aggregate-marker-count must be >= 0")
    if args.run_model is not None and not str(args.run_model).strip():
        raise ValueError("run-model must be a non-empty string")
    if args.run_explain is not None and not str(args.run_explain).strip():
        raise ValueError("run-explain must be a non-empty string")
    if args.run_approach is not None and not str(args.run_approach).strip():
        raise ValueError("run-approach must be a non-empty string")

    run_model = str(args.run_model).strip() if args.run_model is not None else None
    run_explain = str(args.run_explain).strip() if args.run_explain is not None else None
    run_approach = str(args.run_approach).strip() if args.run_approach is not None else None

    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir not found or not a directory: {run_dir}")

    output_root = Path(args.output_root) if args.output_root else None
    summary_paths = _find_summary_files(run_dir)
    if not summary_paths:
        raise RuntimeError(f"No summary.json/summarize.json found under: {run_dir}")

    total = 0
    ok = 0
    skipped = 0
    failed = 0

    do_sample_plots = bool(args.save_sample_plots) and not bool(args.skip_sample_plots)

    if do_sample_plots:
        for summary_path in _iter_targets(summary_paths):
            total += 1
            payload = _read_json(summary_path)
            if payload is None:
                failed += 1
                print(f"[FAILED] invalid json: {summary_path}")
                continue

            if run_model:
                model_name_in_payload = _extract_model_name(summary_path, run_dir, payload)
                if model_name_in_payload != run_model:
                    skipped += 1
                    continue

            if run_explain:
                explain_in_payload = _extract_explain_method(summary_path, run_dir, payload)
                if explain_in_payload != run_explain:
                    skipped += 1
                    continue

            if run_approach:
                approach_in_payload = _extract_approach_folder_name(summary_path, run_dir, payload)
                if approach_in_payload != run_approach:
                    skipped += 1
                    continue

            if not args.allow_non_ok and payload.get("status") not in {None, "ok"}:
                skipped += 1
                print(f"[SKIP] status={payload.get('status')} path={summary_path}")
                continue

            curves = _extract_curves(payload)
            if curves is None:
                skipped += 1
                print(f"[SKIP] no valid curves in: {summary_path}")
                continue

            sample_output_dir = summary_path.parent if output_root is None else (output_root / summary_path.parent.relative_to(run_dir))

            deletion_clean_path = sample_output_dir / args.deletion_clean_name
            deletion_adv_path = sample_output_dir / args.deletion_adv_name
            insertion_clean_path = sample_output_dir / args.insertion_clean_name
            insertion_adv_path = sample_output_dir / args.insertion_adv_name
            deletion_overlay_path = sample_output_dir / args.deletion_overlay_name
            insertion_overlay_path = sample_output_dir / args.insertion_overlay_name

            try:
                _plot_one_curve(curves["clean_del"], deletion_clean_path, args.dpi)
                _plot_one_curve(curves["adv_del"], deletion_adv_path, args.dpi)
                _plot_one_curve(curves["clean_ins"], insertion_clean_path, args.dpi)
                _plot_one_curve(curves["adv_ins"], insertion_adv_path, args.dpi)
                _plot_overlay_curves(curves["clean_del"], curves["adv_del"], deletion_overlay_path, args.dpi)
                _plot_overlay_curves(curves["clean_ins"], curves["adv_ins"], insertion_overlay_path, args.dpi)
                ok += 1
                print(f"[OK] {summary_path.parent}")
                print(f"      - {deletion_clean_path}")
                print(f"      - {deletion_adv_path}")
                print(f"      - {insertion_clean_path}")
                print(f"      - {insertion_adv_path}")
                print(f"      - {deletion_overlay_path}")
                print(f"      - {insertion_overlay_path}")
            except Exception as exc:
                failed += 1
                print(f"[FAILED] {summary_path} error={exc}")

    _aggregate_and_plot(
        run_dir=run_dir,
        summary_paths=summary_paths,
        output_root=output_root,
        dpi=args.dpi,
        allow_non_ok=args.allow_non_ok,
        aggregate_y_min=float(args.aggregate_y_min),
        aggregate_y_max=float(args.aggregate_y_max),
        aggregate_marker_count=int(args.aggregate_marker_count),
        aggregate_points_only=bool(args.aggregate_points_only),
        run_model=run_model,
        run_explain=run_explain,
        run_approach=run_approach,
    )

    print("=== Done ===")
    if not do_sample_plots:
        print(f"total summaries scanned: {len(summary_paths)}")
        print("per-sample plots: disabled")
    else:
        print(f"total summaries: {total}")
        print(f"ok: {ok}")
        print(f"skipped: {skipped}")
        print(f"failed: {failed}")
        print("per-sample plots: enabled")
    if run_model:
        print(f"model filter: {run_model}")
    if run_explain:
        print(f"explain filter: {run_explain}")
    if run_approach:
        print(f"approach filter: {run_approach}")
    print("aggregate plots generated at method/model directories and _method_aggregates/")


if __name__ == "__main__":
    main()
    
    
