import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    i = 0
    n = sorted_vals.size
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1

        avg_rank = (i + 1 + j) * 0.5
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.size != y.size:
        raise ValueError("Spearman inputs must have same number of elements")
    if x.size < 2:
        return float("nan")

    rx = rankdata_average_ties(x)
    ry = rankdata_average_ties(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = math.sqrt(float(np.dot(rx, rx)) * float(np.dot(ry, ry)))
    if denom < 1e-12:
        return float("nan")

    return float(np.dot(rx, ry) / denom)


def load_gray_image_array(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    return np.asarray(image, dtype=np.float64)


def _safe_float(value: object) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _read_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _read_spearman_from_payload(payload: Dict[str, object]) -> Optional[float]:
    direct = _safe_float(payload.get("spearman_best_adv_vs_clean"))
    if direct is not None:
        return direct

    history = payload.get("history_spearman")
    if isinstance(history, list):
        for value in reversed(history):
            val = _safe_float(value)
            if val is not None:
                return val
    return None


def _find_summary_files(run_dir: Path) -> List[Path]:
    summary_paths = list(run_dir.glob("**/summary.json"))
    summary_paths.extend(run_dir.glob("**/summarize.json"))
    return sorted(set(summary_paths))


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
            explain_method = run_segments[1]
            approach_segments = run_segments[2:]
        else:
            explain_method = run_segments[model_idx - 1]
            approach_segments = run_segments[model_idx + 1 :]

        approach = "/".join(approach_segments) if approach_segments else "default"
        return model_name, explain_method, approach

    model_name = run_segments[0]
    explain_method = run_segments[1]
    approach = "/".join(run_segments[2:]) if len(run_segments) > 2 else "default"
    return model_name, explain_method, approach


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
                if len(run_segments) > 2 and run_segments[2].strip():
                    return run_segments[2].strip()
            else:
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


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _fmt_num(value: Optional[float]) -> str:
    if value is None:
        return "nan"
    return f"{float(value):.6f}"


def _normalize_approach_label(approach_name: str) -> str:
    raw = approach_name.strip()
    lower = raw.lower()
    if "insert_oriented" in lower:
        return "insert_oriented"
    if "delete_oriented" in lower:
        return "delete_oriented"
    if "saliency_optimized" in lower:
        return "saliency_optimized"
    if "both_" in lower:
        return raw
    if "mixed_wc=" in lower:
        return raw
    return raw


def _plot_category_bar_chart(category_scores: Dict[str, float], output_path: Path, dpi: int, y_label: str) -> None:
    if not category_scores:
        return

    sorted_items = sorted(category_scores.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [float(item[1]) for item in sorted_items]

    fig_w = max(8.0, 0.8 * len(labels) + 4.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 4.8))
    bars = ax.bar(np.arange(len(labels)), values, color="#c0392b", alpha=0.9)

    ax.set_ylabel(y_label)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.25)

    for rect, val in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() * 0.5,
            rect.get_height() + (0.02 if val >= 0 else -0.06),
            f"{val:.3f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def run_aggregate(
    run_dir: Path,
    output_root: Optional[Path],
    run_model: Optional[str],
    run_explain: Optional[str],
    run_approach: Optional[str],
    allow_non_ok: bool,
    dpi: int,
) -> None:
    summary_paths = _find_summary_files(run_dir)
    if not summary_paths:
        raise FileNotFoundError(f"No summary files found under: {run_dir}")

    approach_values: Dict[str, List[float]] = {}
    method_values: Dict[str, List[float]] = {}
    method_approach_values: Dict[str, Dict[str, List[float]]] = {}
    detailed_rows: List[Tuple[str, str, str, float]] = []

    for summary_path in summary_paths:
        payload = _read_json(summary_path)
        if payload is None:
            continue

        if not allow_non_ok and payload.get("status") not in {None, "ok"}:
            continue

        model_name = _extract_model_name(summary_path, run_dir, payload)
        explain_method = _extract_explain_method(summary_path, run_dir, payload)
        approach_name = _extract_approach_folder_name(summary_path, run_dir, payload)

        if model_name is None or explain_method is None or approach_name is None:
            continue

        if run_model and model_name != run_model:
            continue
        if run_explain and explain_method != run_explain:
            continue
        if run_approach and approach_name != run_approach:
            continue

        spearman_value = _read_spearman_from_payload(payload)
        if spearman_value is None:
            continue

        normalized_approach = _normalize_approach_label(approach_name)
        approach_values.setdefault(normalized_approach, []).append(spearman_value)
        method_values.setdefault(explain_method, []).append(spearman_value)

        method_holder = method_approach_values.setdefault(explain_method, {})
        method_holder.setdefault(normalized_approach, []).append(spearman_value)

        detailed_rows.append((explain_method, model_name, normalized_approach, spearman_value))

    if not detailed_rows:
        raise ValueError("No valid Spearman values found for the selected filters.")

    approach_mean = {
        approach: mean_val
        for approach, mean_val in ((a, _mean(vals)) for a, vals in approach_values.items())
        if mean_val is not None
    }
    method_mean = {
        method: mean_val
        for method, mean_val in ((m, _mean(vals)) for m, vals in method_values.items())
        if mean_val is not None
    }

    out_root = output_root if output_root is not None else run_dir
    out_dir = out_root / "_method_aggregates"
    out_dir.mkdir(parents=True, exist_ok=True)

    approach_txt_path = out_dir / "summary_spearman_by_approach.txt"
    with open(approach_txt_path, "w", encoding="utf-8") as f:
        f.write("approach & spearman\n")
        for approach_name in sorted(approach_mean.keys()):
            f.write(f"{approach_name} & {_fmt_num(approach_mean[approach_name])}\n")

    method_txt_path = out_dir / "summary_spearman_by_explain_method.txt"
    with open(method_txt_path, "w", encoding="utf-8") as f:
        f.write("explain_method & spearman\n")
        for explain_method in sorted(method_mean.keys()):
            f.write(f"{explain_method} & {_fmt_num(method_mean[explain_method])}\n")

    detailed_txt_path = out_dir / "summary_spearman_by_model_approach.txt"
    with open(detailed_txt_path, "w", encoding="utf-8") as f:
        f.write("explain_method & model_name & approach & spearman\n")
        for explain_method, model_name, approach_name, spearman_value in sorted(detailed_rows):
            f.write(
                f"{explain_method} & {model_name} & {approach_name} & {_fmt_num(spearman_value)}\n"
            )

    _plot_category_bar_chart(
        approach_mean,
        out_dir / "spearman_compare_approaches_bar.png",
        dpi=dpi,
        y_label="Mean Spearman",
    )

    for explain_method, approach_block in sorted(method_approach_values.items()):
        per_method_approach_mean = {
            approach: mean_val
            for approach, mean_val in ((a, _mean(vals)) for a, vals in approach_block.items())
            if mean_val is not None
        }
        if not per_method_approach_mean:
            continue
        safe_method = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in explain_method)
        _plot_category_bar_chart(
            per_method_approach_mean,
            out_dir / f"spearman_compare_approaches__{safe_method}.png",
            dpi=dpi,
            y_label="Mean Spearman",
        )

    print(f"summary_approach_txt: {approach_txt_path}")
    print(f"summary_method_txt: {method_txt_path}")
    print(f"detail_txt: {detailed_txt_path}")
    print(f"chart: {out_dir / 'spearman_compare_approaches_bar.png'}")
    print(f"num_methods: {len(method_mean)}")
    print(f"num_approaches: {len(approach_mean)}")
    print(f"num_samples: {len(detailed_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spearman utilities for mask pairs and run-level aggregation")
    parser.add_argument("--img1", type=str, default=None, help="Path to first image (PNG mask)")
    parser.add_argument("--img2", type=str, default=None, help="Path to second image (PNG mask)")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run folder path for summary aggregation mode (scans summary.json recursively).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional root to write aggregate outputs. Default: inside run-dir/_method_aggregates.",
    )
    parser.add_argument("--run-model", type=str, default=None, help="Optional model filter (exact match).")
    parser.add_argument("--run-explain", type=str, default=None, help="Optional explain method filter (exact match).")
    parser.add_argument("--run-approach", type=str, default=None, help="Optional approach filter (exact match).")
    parser.add_argument(
        "--allow-non-ok",
        action="store_true",
        help="Include summary files whose status is not ok.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI for output chart in aggregate mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        output_root = Path(args.output_root) if args.output_root else None
        run_aggregate(
            run_dir=run_dir,
            output_root=output_root,
            run_model=args.run_model,
            run_explain=args.run_explain,
            run_approach=args.run_approach,
            allow_non_ok=bool(args.allow_non_ok),
            dpi=int(args.dpi),
        )
        return

    if not args.img1 or not args.img2:
        raise ValueError("Pair mode requires both --img1 and --img2, or use --run-dir for aggregate mode.")

    img1_path = Path(args.img1)
    img2_path = Path(args.img2)

    if not img1_path.exists():
        raise FileNotFoundError(f"Image not found: {img1_path}")
    if not img2_path.exists():
        raise FileNotFoundError(f"Image not found: {img2_path}")

    img1 = load_gray_image_array(img1_path)
    img2 = load_gray_image_array(img2_path)

    if img1.shape != img2.shape:
        raise ValueError(
            f"Image shapes differ: img1={img1.shape}, img2={img2.shape}. "
            "Please resize/crop to the same size before computing Spearman."
        )

    corr = spearman_rank_corr(img1, img2)
    print(f"spearman_corr: {corr:.10f}")


if __name__ == "__main__":
    main()
