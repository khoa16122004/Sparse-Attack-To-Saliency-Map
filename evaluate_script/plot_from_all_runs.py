import argparse
import json
from pathlib import Path
from typing import List

from statis_results import (
    _build_grouped_fourway_curves,
    _build_overall_compare_init,
    _build_overall_compare_loss,
    _build_overall_grouped_fourway,
    _build_pair_curves_init,
    _build_pair_curves_loss,
    _is_close,
    _normalize_algorithm,
    _parse_optional_float_arg,
    _parse_weight_pairs,
    _plot_grouped_fourway,
    _plot_overall_compare_init,
    _plot_overall_compare_loss,
    _plot_overall_grouped_fourway,
    _plot_pair_curves_init,
    _plot_pair_curves_loss,
    _runstats_from_dict,
    _save_json,
    RunStats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load precomputed all_runs.json, filter, aggregate, and generate plots/JSON outputs.",
    )
    parser.add_argument(
        "--all-runs-json",
        type=str,
        required=True,
        help="Path to precomputed all_runs.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluate_script/stats_outputs",
        help="Directory to save filtered summaries and plots",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="weighted_sum_ga",
        help="Keep only this algorithm. Use 'all' to keep all.",
    )
    parser.add_argument(
        "--explain-method",
        type=str,
        default="simple_gradient",
        help="Keep only this explain method. Use 'all' to keep all.",
    )
    parser.add_argument(
        "--w-m",
        type=str,
        default="all",
        help="Filter by w_m. Use numeric value or 'all'.",
    )
    parser.add_argument(
        "--w-c",
        type=str,
        default="all",
        help="Backward-compatible alias for --w-s.",
    )
    parser.add_argument(
        "--w-s",
        type=str,
        default="all",
        help="Filter by w_s. Use numeric value or 'all'.",
    )
    parser.add_argument(
        "--weight-pairs",
        type=str,
        default=None,
        help="Comma-separated wm:ws pairs, e.g. '1:0,0.5:0.5'.",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Generate PNG plots",
    )
    return parser.parse_args()


def _load_runs(path: Path) -> List[RunStats]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("all_runs.json must contain a list")

    return [_runstats_from_dict(item) for item in raw if isinstance(item, dict)]


def main() -> None:
    args = parse_args()

    all_runs_path = Path(args.all_runs_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not all_runs_path.exists():
        raise FileNotFoundError(f"all_runs JSON not found: {all_runs_path}")

    all_runs = _load_runs(all_runs_path)
    if not all_runs:
        raise ValueError("No run found in provided all_runs JSON")

    filtered = all_runs
    if args.algorithm.lower() != "all":
        target_algo = _normalize_algorithm(args.algorithm)
        filtered = [r for r in filtered if r.algorithm == target_algo]

    if args.explain_method.lower() != "all":
        filtered = [r for r in filtered if r.explain_method == args.explain_method]

    target_w_m = _parse_optional_float_arg(args.w_m, "--w-m")
    raw_w_s = args.w_s if str(args.w_s).strip().lower() != "all" else args.w_c
    target_w_s = _parse_optional_float_arg(raw_w_s, "--w-s")
    target_pairs = _parse_weight_pairs(args.weight_pairs)

    if target_w_m is not None:
        filtered = [r for r in filtered if _is_close(r.w_m, target_w_m)]
    if target_w_s is not None:
        filtered = [r for r in filtered if _is_close(r.w_s, target_w_s)]
    if target_pairs:
        filtered = [
            r
            for r in filtered
            if any(_is_close(r.w_m, wm) and _is_close(r.w_s, ws) for wm, ws in target_pairs)
        ]

    if not filtered:
        raise ValueError("No run left after filtering")

    loss_pairs = _build_pair_curves_loss(filtered)
    init_pairs = _build_pair_curves_init(filtered)
    grouped_fourway = _build_grouped_fourway_curves(filtered)

    overall_compare_loss = _build_overall_compare_loss(loss_pairs)
    overall_compare_init = _build_overall_compare_init(init_pairs)
    overall_grouped_fourway = _build_overall_grouped_fourway(grouped_fourway)

    _save_json(output_dir / "all_runs.json", [r.__dict__ | {"run_dir": str(r.run_dir)} for r in filtered])
    _save_json(output_dir / "compare_loss_curves.json", loss_pairs)
    _save_json(output_dir / "compare_init_curves.json", init_pairs)
    _save_json(output_dir / "compare_loss_curves_overall.json", overall_compare_loss)
    _save_json(output_dir / "compare_init_curves_overall.json", overall_compare_init)
    _save_json(output_dir / "compare_grouped_fourway_curves.json", grouped_fourway)
    _save_json(output_dir / "compare_grouped_fourway_overall.json", overall_grouped_fourway)

    if args.make_plots:
        _plot_pair_curves_loss(loss_pairs, output_dir)
        _plot_pair_curves_init(init_pairs, output_dir)
        _plot_overall_compare_loss(overall_compare_loss, output_dir)
        _plot_overall_compare_init(overall_compare_init, output_dir)
        _plot_grouped_fourway(grouped_fourway, output_dir)
        _plot_overall_grouped_fourway(overall_grouped_fourway, output_dir)

    print(f"Loaded all_runs entries: {len(all_runs)}")
    print(f"Filtered runs: {len(filtered)}")
    print(f"Loss pairs: {len(loss_pairs)}")
    print(f"Init pairs: {len(init_pairs)}")
    print(f"Group4 sets: {len(grouped_fourway)}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
