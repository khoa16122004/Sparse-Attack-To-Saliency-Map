import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


def _find_summary_files(run_dir: Path) -> List[Path]:
    summary_paths = list(run_dir.glob("**/summary.json"))
    summary_paths.extend(run_dir.glob("**/summarize.json"))
    # Keep deterministic order and remove duplicates.
    return sorted(set(summary_paths))


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

    if clean_curve:
        x_clean = np.linspace(0.0, 1.0, num=len(clean_curve), endpoint=True)
        y_clean = np.asarray(clean_curve, dtype=float)
        ax.plot(x_clean, y_clean, linewidth=1.8, color="#1f77b4")
        ax.fill_between(x_clean, y_clean, 0.0, color="#1f77b4", alpha=0.22)

    if adv_curve:
        x_adv = np.linspace(0.0, 1.0, num=len(adv_curve), endpoint=True)
        y_adv = np.asarray(adv_curve, dtype=float)
        ax.plot(x_adv, y_adv, linewidth=1.8, color="#d62728")
        ax.fill_between(x_adv, y_adv, 0.0, color="#d62728", alpha=0.22)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


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
    args = parser.parse_args()

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

    for summary_path in _iter_targets(summary_paths):
        total += 1
        payload = _read_json(summary_path)
        if payload is None:
            failed += 1
            print(f"[FAILED] invalid json: {summary_path}")
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

    print("=== Done ===")
    print(f"total summaries: {total}")
    print(f"ok: {ok}")
    print(f"skipped: {skipped}")
    print(f"failed: {failed}")


if __name__ == "__main__":
    main()
