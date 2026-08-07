import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunCurves:
    run_dir: Path
    model_name: Optional[str]
    wm: Optional[float]
    ws: Optional[float]
    adv_del_mean: List[float]
    adv_ins_mean: List[float]
    adv_imd_mean: List[float]
    clean_del_curves: List[List[float]]
    clean_ins_curves: List[List[float]]
    sample_count: int


def _safe_float(raw: str) -> Optional[float]:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _parse_weights_from_run_name(run_name: str) -> Tuple[Optional[float], Optional[float]]:
    wm = None
    ws = None

    wm_match = re.search(r"(?:^|__)wm-([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", run_name)
    ws_match = re.search(r"(?:^|__)ws-([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", run_name)

    if wm_match:
        wm = _safe_float(wm_match.group(1))
    if ws_match:
        ws = _safe_float(ws_match.group(1))

    if wm is not None and ws is None:
        ws = 1.0 - wm
    if ws is not None and wm is None:
        wm = 1.0 - ws

    return wm, ws


def _curve_mean_with_last_padding(curves: Sequence[Sequence[float]]) -> List[float]:
    valid = [list(curve) for curve in curves if curve]
    if not valid:
        return []

    max_len = max(len(curve) for curve in valid)
    out: List[float] = []

    for i in range(max_len):
        values = [curve[i] if i < len(curve) else curve[-1] for curve in valid]
        out.append(float(sum(values) / len(values)))

    return out


def _curve_diff_with_last_padding(
    minuend_curve: Sequence[float],
    subtrahend_curve: Sequence[float],
) -> List[float]:
    if not minuend_curve or not subtrahend_curve:
        return []

    left = list(minuend_curve)
    right = list(subtrahend_curve)
    max_len = max(len(left), len(right))
    out: List[float] = []

    for i in range(max_len):
        lval = left[i] if i < len(left) else left[-1]
        rval = right[i] if i < len(right) else right[-1]
        out.append(float(lval - rval))

    return out


def _has_summary_files(path: Path) -> bool:
    return any(path.glob("*/*/summary.json")) or any(path.glob("*/*/summarize.json"))


def _find_run_dirs(root_dir: Path) -> List[Path]:
    if _has_summary_files(root_dir):
        return [root_dir]

    run_dirs: List[Path] = []
    for child in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        if _has_summary_files(child):
            run_dirs.append(child)

    return run_dirs


def _find_model_dirs(root_dir: Path) -> List[Path]:
    model_dirs: List[Path] = []
    for child in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        if _find_run_dirs(child):
            model_dirs.append(child)
    return model_dirs


def _load_summary_paths(run_dir: Path) -> List[Path]:
    paths = list(run_dir.glob("*/*/summary.json")) + list(run_dir.glob("*/*/summarize.json"))
    return sorted(set(paths))


def _read_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


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


def _collect_run_curves(run_dir: Path, model_name: Optional[str] = None) -> Optional[RunCurves]:
    summary_paths = _load_summary_paths(run_dir)
    if not summary_paths:
        return None

    adv_del_curves: List[List[float]] = []
    adv_ins_curves: List[List[float]] = []
    clean_del_curves: List[List[float]] = []
    clean_ins_curves: List[List[float]] = []

    for summary_path in summary_paths:
        payload = _read_json(summary_path)
        if payload is None:
            continue

        if payload.get("status") != "ok":
            continue

        causal_block = payload.get("causual")
        if not isinstance(causal_block, dict):
            causal_block = payload.get("causal")
        if not isinstance(causal_block, dict):
            continue

        adv_del = _as_float_list(causal_block.get("del"))
        adv_ins = _as_float_list(causal_block.get("ins"))
        clean_del = _as_float_list(causal_block.get("clean_del"))
        clean_ins = _as_float_list(causal_block.get("clean_ins"))

        if adv_del:
            adv_del_curves.append(adv_del)
        if adv_ins:
            adv_ins_curves.append(adv_ins)
        if clean_del:
            clean_del_curves.append(clean_del)
        if clean_ins:
            clean_ins_curves.append(clean_ins)

    if not adv_del_curves and not adv_ins_curves:
        return None

    wm, ws = _parse_weights_from_run_name(run_dir.name)

    return RunCurves(
        run_dir=run_dir,
        model_name=model_name,
        wm=wm,
        ws=ws,
        adv_del_mean=_curve_mean_with_last_padding(adv_del_curves),
        adv_ins_mean=_curve_mean_with_last_padding(adv_ins_curves),
        adv_imd_mean=_curve_diff_with_last_padding(
            _curve_mean_with_last_padding(adv_ins_curves),
            _curve_mean_with_last_padding(adv_del_curves),
        ),
        clean_del_curves=clean_del_curves,
        clean_ins_curves=clean_ins_curves,
        sample_count=max(len(adv_del_curves), len(adv_ins_curves)),
    )


def _group_key_for_cross_model_average(run: RunCurves) -> Tuple[str, object, object]:
    if run.wm is not None or run.ws is not None:
        wm = None if run.wm is None else round(float(run.wm), 12)
        ws = None if run.ws is None else round(float(run.ws), 12)
        return ("weights", wm, ws)
    return ("name", run.run_dir.name, None)


def _aggregate_across_models(run_curves: List[RunCurves]) -> List[RunCurves]:
    grouped: Dict[Tuple[str, object, object], List[RunCurves]] = {}
    for run in run_curves:
        key = _group_key_for_cross_model_average(run)
        grouped.setdefault(key, []).append(run)

    merged: List[RunCurves] = []
    for key, group in grouped.items():
        wm = group[0].wm
        ws = group[0].ws
        label = group[0].run_dir.name
        if key[0] == "weights":
            wm_txt = "na" if wm is None else f"{wm:g}"
            ws_txt = "na" if ws is None else f"{ws:g}"
            label = f"avg_models__wm-{wm_txt}__ws-{ws_txt}"

        adv_del_curves = [g.adv_del_mean for g in group if g.adv_del_mean]
        adv_ins_curves = [g.adv_ins_mean for g in group if g.adv_ins_mean]

        adv_del_mean = _curve_mean_with_last_padding(adv_del_curves)
        adv_ins_mean = _curve_mean_with_last_padding(adv_ins_curves)
        adv_imd_mean = _curve_diff_with_last_padding(adv_ins_mean, adv_del_mean)

        clean_del_all: List[List[float]] = []
        clean_ins_all: List[List[float]] = []
        for g in group:
            clean_del_all.extend(g.clean_del_curves)
            clean_ins_all.extend(g.clean_ins_curves)

        merged.append(
            RunCurves(
                run_dir=Path(label),
                model_name=None,
                wm=wm,
                ws=ws,
                adv_del_mean=adv_del_mean,
                adv_ins_mean=adv_ins_mean,
                adv_imd_mean=adv_imd_mean,
                clean_del_curves=clean_del_all,
                clean_ins_curves=clean_ins_all,
                sample_count=sum(g.sample_count for g in group),
            )
        )

    return merged


def _format_lambda_label(wm: Optional[float], ws: Optional[float], sample_count: int) -> str:
    if wm is None:
        return r"$\lambda=?$"
    return rf"$\lambda={wm:g}$"


def _sort_key(run: RunCurves) -> Tuple[float, float, str]:
    wm = run.wm if run.wm is not None else 1e9
    ws = run.ws if run.ws is not None else 1e9
    return wm, ws, run.run_dir.name


def _style_for_run(run: RunCurves, index: int) -> Dict[str, object]:
    # Match the visual prototype: fixed style by lambda when available.
    style_by_lambda: List[Tuple[float, Dict[str, object]]] = [
        (0.0, {"color": "#d62728", "marker": "o", "linestyle": "-"}),
        (0.2, {"color": "#1f77b4", "marker": "s", "linestyle": "--"}),
        (0.5, {"color": "#2ca02c", "marker": "D", "linestyle": "-."}),
        (0.8, {"color": "#9467bd", "marker": "^", "linestyle": ":"}),
        (1.0, {"color": "#ff7f0e", "marker": "v", "linestyle": "--"}),
    ]

    if run.wm is not None:
        for lam, style in style_by_lambda:
            if abs(float(run.wm) - lam) <= 1e-9:
                return style

    fallback = [
        {"color": "#d62728", "marker": "o", "linestyle": "-"},
        {"color": "#1f77b4", "marker": "s", "linestyle": "--"},
        {"color": "#2ca02c", "marker": "D", "linestyle": "-."},
        {"color": "#9467bd", "marker": "^", "linestyle": ":"},
        {"color": "#ff7f0e", "marker": "v", "linestyle": "--"},
        {"color": "#8c564b", "marker": "P", "linestyle": "-"},
    ]
    return fallback[index % len(fallback)]


def _markevery_from_count(curve_len: int, marker_count: int) -> int:
    if curve_len <= 0:
        return 1
    marker_count = max(1, int(marker_count))
    if marker_count == 1:
        return max(1, curve_len - 1)
    return max(1, int(round((curve_len - 1) / marker_count)))


def _annotate_curve_labels(
    ax,
    labels_data: List[Tuple[float, float, str, str]],
    text_size: float,
    x_offset_ratio: float,
) -> None:
    if not labels_data:
        return

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    y_span = max(1e-9, y_max - y_min)
    min_gap = 0.05 * y_span

    sorted_data = sorted(labels_data, key=lambda item: item[1])
    adjusted: List[Tuple[float, float, str, str]] = []
    prev_y = None
    for x, y, label, color in sorted_data:
        y_adj = y
        if prev_y is not None and y_adj < prev_y + min_gap:
            y_adj = prev_y + min_gap
        adjusted.append((x, y_adj, label, color))
        prev_y = y_adj

    if adjusted and adjusted[-1][1] > y_max:
        shift = adjusted[-1][1] - y_max
        adjusted = [(x, y - shift, label, color) for x, y, label, color in adjusted]

    x_offset = x_offset_ratio * (x_max - x_min)
    for x, y, label, color in adjusted:
        ax.text(x + x_offset, y, label, color=color, fontsize=text_size, va="center", ha="left")


def _plot_side_by_side(
    run_curves: List[RunCurves],
    clean_del_mean: List[float],
    clean_ins_mean: List[float],
    clean_imd_mean: List[float],
    output_path: Path,
    title: str,
    marker_count: int,
    fig_scale: float,
    zoom_x: float,
    zoom_y: float,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_scale = max(0.5, float(fig_scale))
    zoom_x = max(1.0, float(zoom_x))
    zoom_y = max(1.0, float(zoom_y))
    dpi = max(72, int(dpi))
    base_w, base_h = 22.0, 6.8
    fig, axes = plt.subplots(1, 3, figsize=(base_w * fig_scale, base_h * fig_scale), sharey=False)
    ax_del, ax_ins, ax_imd = axes
    line_w = 3.1 * fig_scale
    marker_sz = 5.2 * fig_scale
    clean_line_w = 3.6 * fig_scale
    clean_marker_sz = 6.0 * fig_scale
    tick_size = 15.0 * fig_scale
    label_text_size = 14.5 * fig_scale

    sorted_runs = sorted(run_curves, key=_sort_key)
    for i, run in enumerate(sorted_runs):
        style = _style_for_run(run, i)
        label = _format_lambda_label(run.wm, run.ws, run.sample_count)
        markevery_del = _markevery_from_count(len(run.adv_del_mean), marker_count)
        markevery_ins = _markevery_from_count(len(run.adv_ins_mean), marker_count)

        if run.adv_del_mean:
            x_del = np.arange(len(run.adv_del_mean), dtype=np.int32)
            ax_del.plot(
                x_del,
                run.adv_del_mean,
                linewidth=line_w,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=marker_sz,
                markevery=markevery_del,
            )

        if run.adv_ins_mean:
            x_ins = np.arange(len(run.adv_ins_mean), dtype=np.int32)
            ax_ins.plot(
                x_ins,
                run.adv_ins_mean,
                linewidth=line_w,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=marker_sz,
                markevery=markevery_ins,
            )

        if run.adv_imd_mean:
            x_imd = np.arange(len(run.adv_imd_mean), dtype=np.int32)
            ax_imd.plot(
                x_imd,
                run.adv_imd_mean,
                linewidth=line_w,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=marker_sz,
                markevery=_markevery_from_count(len(run.adv_imd_mean), marker_count),
            )

    if clean_del_mean:
        x_clean_del = np.arange(len(clean_del_mean), dtype=np.int32)
        ax_del.plot(
            x_clean_del,
            clean_del_mean,
            color="black",
            linestyle="--",
            linewidth=clean_line_w,
            marker="x",
            markersize=clean_marker_sz,
            markevery=_markevery_from_count(len(clean_del_mean), marker_count),
            label="clean",
        )

    if clean_ins_mean:
        x_clean_ins = np.arange(len(clean_ins_mean), dtype=np.int32)
        ax_ins.plot(
            x_clean_ins,
            clean_ins_mean,
            color="black",
            linestyle="--",
            linewidth=clean_line_w,
            marker="x",
            markersize=clean_marker_sz,
            markevery=_markevery_from_count(len(clean_ins_mean), marker_count),
        )

    if clean_imd_mean:
        x_clean_imd = np.arange(len(clean_imd_mean), dtype=np.int32)
        ax_imd.plot(
            x_clean_imd,
            clean_imd_mean,
            color="black",
            linestyle="--",
            linewidth=clean_line_w,
            marker="x",
            markersize=clean_marker_sz,
            markevery=_markevery_from_count(len(clean_imd_mean), marker_count),
        )

    ax_del.set_title("")
    ax_del.set_xlabel("")
    ax_del.set_ylabel("")
    ax_del.tick_params(axis="both", labelsize=tick_size)

    ax_ins.set_title("")
    ax_ins.set_xlabel("")
    ax_ins.set_ylabel("")
    ax_ins.tick_params(axis="both", labelsize=tick_size)

    ax_imd.set_title("")
    ax_imd.set_xlabel("")
    ax_imd.set_ylabel("")
    ax_imd.tick_params(axis="both", labelsize=tick_size)

    for axis in (ax_del, ax_ins, ax_imd):
        x_left, x_right = axis.get_xlim()
        x_span = x_right - x_left if x_right > x_left else 1.0
        x_mid = 0.5 * (x_left + x_right)
        x_new_span = x_span / zoom_x
        axis.set_xlim(x_mid - 0.5 * x_new_span, x_mid + 0.5 * x_new_span + 0.16 * x_new_span)

        y_bottom, y_top = axis.get_ylim()
        y_span = y_top - y_bottom if y_top > y_bottom else 1.0
        y_mid = 0.5 * (y_bottom + y_top)
        y_new_span = y_span / zoom_y
        axis.set_ylim(y_mid - 0.5 * y_new_span, y_mid + 0.5 * y_new_span)

    legend_handles, legend_labels = ax_del.get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=max(2, len(legend_labels)),
        fontsize=label_text_size,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    if title:
        fig.suptitle(title, fontsize=20.0 * fig_scale)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.9])
    else:
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _resolve_output_path(root_dir: Path, output_arg: Optional[str]) -> Path:
    default_name = "avg_del_ins_imd_curves.png"
    if output_arg is None:
        return root_dir / default_name

    raw = str(output_arg).strip()
    if raw in {"", ".", ".."}:
        return Path(raw) / default_name

    candidate = Path(output_arg)
    if candidate.exists() and candidate.is_dir():
        return candidate / default_name

    if candidate.suffix:
        return candidate

    if raw.endswith("/") or raw.endswith("\\"):
        return candidate / default_name

    return candidate.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan run folders, compute mean deletion/insertion curves per wm/ws, "
            "plot deletion/insertion/IMD in one prototype-style figure."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to a model folder (contains run dirs) or a single run dir.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: <root>/avg_del_ins_curves.png)",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional output JSON with aggregation metadata.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Figure title",
    )
    parser.add_argument(
        "--marker-count",
        type=int,
        default=10,
        help="Approximate number of markers to show on each curve.",
    )
    parser.add_argument(
        "--fig-scale",
        type=float,
        default=2.8,
        help="Overall scale for figure size and text (e.g. 1.0, 1.5, 2.0).",
    )
    parser.add_argument(
        "--zoom-x",
        type=float,
        default=1.0,
        help="Zoom factor for x-axis (>1 zooms in).",
    )
    parser.add_argument(
        "--zoom-y",
        type=float,
        default=1.0,
        help="Zoom factor for y-axis (>1 zooms in).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=340,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--average-models",
        action="store_true",
        help="If root contains multiple model folders, average corresponding runs across models.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root)
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"root not found or not a directory: {root_dir}")

    model_dirs = _find_model_dirs(root_dir)
    use_cross_model_average = bool(model_dirs) and args.average_models

    run_curves_raw: List[RunCurves] = []
    all_clean_del_curves: List[List[float]] = []
    all_clean_ins_curves: List[List[float]] = []

    if model_dirs:
        for model_dir in model_dirs:
            run_dirs = _find_run_dirs(model_dir)
            for run_dir in run_dirs:
                packed = _collect_run_curves(run_dir, model_name=model_dir.name)
                if packed is None:
                    continue
                run_curves_raw.append(packed)
                all_clean_del_curves.extend(packed.clean_del_curves)
                all_clean_ins_curves.extend(packed.clean_ins_curves)
    else:
        run_dirs = _find_run_dirs(root_dir)
        for run_dir in run_dirs:
            packed = _collect_run_curves(run_dir, model_name=None)
            if packed is None:
                continue
            run_curves_raw.append(packed)
            all_clean_del_curves.extend(packed.clean_del_curves)
            all_clean_ins_curves.extend(packed.clean_ins_curves)

    if not run_curves_raw:
        raise RuntimeError(f"No valid curves found under: {root_dir}")

    if use_cross_model_average:
        run_curves = _aggregate_across_models(run_curves_raw)
    else:
        run_curves = run_curves_raw

    clean_del_mean = _curve_mean_with_last_padding(all_clean_del_curves)
    clean_ins_mean = _curve_mean_with_last_padding(all_clean_ins_curves)
    clean_imd_mean = _curve_diff_with_last_padding(clean_ins_mean, clean_del_mean)

    output_path = _resolve_output_path(root_dir=root_dir, output_arg=args.output)
    _plot_side_by_side(
        run_curves=run_curves,
        clean_del_mean=clean_del_mean,
        clean_ins_mean=clean_ins_mean,
        clean_imd_mean=clean_imd_mean,
        output_path=output_path,
        title=args.title,
        marker_count=args.marker_count,
        fig_scale=args.fig_scale,
        zoom_x=args.zoom_x,
        zoom_y=args.zoom_y,
        dpi=args.dpi,
    )

    metadata = {
        "root": str(root_dir),
        "output_image": str(output_path),
        "average_models": bool(use_cross_model_average),
        "model_dirs": [p.name for p in model_dirs],
        "num_runs": len(run_curves),
        "runs": [
            {
                "run_dir": str(r.run_dir),
                "model_name": r.model_name,
                "wm": r.wm,
                "ws": r.ws,
                "sample_count": r.sample_count,
                "del_len": len(r.adv_del_mean),
                "ins_len": len(r.adv_ins_mean),
            }
            for r in sorted(run_curves, key=_sort_key)
        ],
        "clean_del_len": len(clean_del_mean),
        "clean_ins_len": len(clean_ins_mean),
        "clean_imd_len": len(clean_imd_mean),
        "marker_count": int(args.marker_count),
        "fig_scale": float(args.fig_scale),
        "zoom_x": float(args.zoom_x),
        "zoom_y": float(args.zoom_y),
        "dpi": int(args.dpi),
    }

    if args.summary_json:
        metadata_path = Path(args.summary_json)
        if metadata_path.exists() and metadata_path.is_dir():
            metadata_path = metadata_path / "avg_del_ins_imd_curves.json"
        elif not metadata_path.suffix:
            metadata_path = metadata_path / "avg_del_ins_imd_curves.json"
    else:
        metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved figure: {output_path}")
    print(f"[OK] Saved metadata: {metadata_path}")
    print(f"[INFO] Runs plotted: {len(run_curves)}")
    for run in sorted(run_curves, key=_sort_key):
        print(
            "[INFO]"
            f" wm={run.wm} ws={run.ws}"
            f" run={run.run_dir.name}"
        )


if __name__ == "__main__":
    main()
