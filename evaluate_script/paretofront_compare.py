import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _safe_float(raw: str) -> Optional[float]:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _is_close(a: Optional[float], b: float, tol: float = 1e-9) -> bool:
    if a is None:
        return False
    return abs(float(a) - float(b)) <= tol


@dataclass
class RunMeta:
    run_dir: Path
    algorithm: str
    strategy: str
    explain_method: str
    eps: int
    w_m: Optional[float]
    w_s: Optional[float]
    loss_type: str


def _parse_approach(approach: str) -> Optional[RunMeta]:
    fields: Dict[str, object] = {
        "strategy": "unknown",
        "eps": None,
        "explain_method": "unknown",
        "algorithm": "ga",
        "loss_type": "margin_loss",
        "w_m": None,
        "w_s": None,
    }

    for token in str(approach).split("__"):
        if token.startswith("strategy-"):
            fields["strategy"] = token[len("strategy-") :]
        elif token.startswith("eps-"):
            eps_raw = _safe_float(token[len("eps-") :])
            if eps_raw is not None:
                fields["eps"] = int(eps_raw)
        elif token.startswith("exp-"):
            fields["explain_method"] = token[len("exp-") :]
        elif token.startswith("algo-"):
            algo_name = token[len("algo-") :].strip().lower()
            if algo_name == "nsgaii":
                fields["algorithm"] = "nsgaii"
        elif token.startswith("wm-"):
            fields["w_m"] = _safe_float(token[len("wm-") :])
        elif token.startswith("w_m-"):
            fields["w_m"] = _safe_float(token[len("w_m-") :])
        elif token.startswith("ws-"):
            fields["w_s"] = _safe_float(token[len("ws-") :])
        elif token.startswith("w_s-"):
            fields["w_s"] = _safe_float(token[len("w_s-") :])
        elif token.startswith("fit-negative_cross_entropy_saliency"):
            fields["loss_type"] = "negative_cross_entropy_saliency"

    if fields["eps"] is None:
        return None

    return RunMeta(
        run_dir=Path("."),
        algorithm=str(fields["algorithm"]),
        strategy=str(fields["strategy"]),
        explain_method=str(fields["explain_method"]),
        eps=int(fields["eps"]),
        w_m=fields["w_m"],
        w_s=fields["w_s"],
        loss_type=str(fields["loss_type"]),
    )


def _parse_weight_pairs(raw: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid weight pair: {part}. Expected wm:ws")
        left, right = part.split(":", 1)
        wm = _safe_float(left.strip())
        ws = _safe_float(right.strip())
        if wm is None or ws is None:
            raise ValueError(f"Invalid weight pair: {part}. wm/ws must be numeric")
        pairs.append((wm, ws))

    if not pairs:
        raise ValueError("At least one weight pair is required")
    return pairs


def _read_front(path: Path) -> Optional[List[Tuple[float, float]]]:
    if not path.exists():
        return None

    points: List[Tuple[float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = re.split(r"[\s,]+", line)
            if len(parts) < 2:
                continue
            x = _safe_float(parts[0])
            y = _safe_float(parts[1])
            if x is None or y is None:
                continue
            points.append((x, y))

    if not points:
        return None

    return sorted(points, key=lambda p: p[0])


def _sample_id_from_front_path(run_dir: Path, front_path: Path) -> str:
    # Expected shape: run_dir/class_name/image_name/non_dominated_front_scores.txt
    rel = front_path.relative_to(run_dir)
    if len(rel.parts) >= 3:
        return f"{rel.parts[0]}/{rel.parts[1]}"
    return str(rel.parent)


def _collect_run_fronts(run_dir: Path) -> Dict[str, Path]:
    fronts: Dict[str, Path] = {}
    for front_path in run_dir.glob("*/*/non_dominated_front_scores.txt"):
        sample_id = _sample_id_from_front_path(run_dir=run_dir, front_path=front_path)
        fronts[sample_id] = front_path
    return fronts


def _collect_run_histories(run_dir: Path) -> Dict[str, Path]:
    histories: Dict[str, Path] = {}
    for history_path in run_dir.glob("*/*/history_scores.txt"):
        sample_id = _sample_id_from_front_path(run_dir=run_dir, front_path=history_path)
        histories[sample_id] = history_path
    return histories


def _read_history(path: Path) -> Optional[Tuple[List[float], List[float]]]:
    if not path.exists():
        return None

    objective: List[float] = []
    saliency: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = re.split(r"[\s,]+", line)
            if len(parts) < 2:
                continue
            obj = _safe_float(parts[0])
            sal = _safe_float(parts[1])
            if obj is None or sal is None:
                continue
            objective.append(obj)
            saliency.append(sal)

    if not objective:
        return None
    return objective, saliency


def _curve_mean_with_last_padding(curves: List[List[float]]) -> List[float]:
    valid = [c for c in curves if c]
    if not valid:
        return []

    max_len = max(len(c) for c in valid)
    out: List[float] = []
    for i in range(max_len):
        values = [curve[i] if i < len(curve) else curve[-1] for curve in valid]
        out.append(sum(values) / len(values))
    return out


def _find_matching_runs(
    root: Path,
    expected_algorithm: str,
    strategy: str,
    explain_method: str,
    eps_list: List[int],
    loss_type: str,
    weight_pairs: Optional[List[Tuple[float, float]]] = None,
) -> Dict[Tuple[int, str], RunMeta]:
    selected: Dict[Tuple[int, str], RunMeta] = {}
    if not root.exists():
        return selected

    eps_set = set(eps_list)
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        parsed = _parse_approach(run_dir.name)
        if parsed is None:
            continue

        parsed.run_dir = run_dir
        if parsed.algorithm != expected_algorithm:
            continue
        if parsed.strategy != strategy:
            continue
        if parsed.explain_method != explain_method:
            continue
        if parsed.eps not in eps_set:
            continue
        if loss_type != "all" and parsed.loss_type != loss_type:
            continue

        if weight_pairs is not None:
            matched_pair: Optional[Tuple[float, float]] = None
            for wm, ws in weight_pairs:
                if _is_close(parsed.w_m, wm) and _is_close(parsed.w_s, ws):
                    matched_pair = (wm, ws)
                    break
            if matched_pair is None:
                continue
            key = (parsed.eps, f"ga_{matched_pair[0]}_{matched_pair[1]}")
        else:
            key = (parsed.eps, "nsgaii")

        selected[key] = parsed

    return selected


def _make_label(key: str) -> str:
    if key == "nsgaii":
        return "NSGAII"
    if key.startswith("ga_"):
        _, wm, ws = key.split("_", 2)
        return f"GA wm={wm} ws={ws}"
    return key


def _plot_one_sample(
    sample_id: str,
    eps: int,
    key_to_points: Dict[str, List[Tuple[float, float]]],
    output_path: Path,
) -> None:
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        print("[WARN] matplotlib not installed, skip plotting")
        return

    color_map = {
        "ga_1.0_0.0": "tab:red",
        "ga_0.5_0.5": "tab:blue",
        "ga_0.0_1.0": "tab:green",
        "nsgaii": "tab:orange",
    }

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))
    for key, points in key_to_points.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=color_map.get(key),
            label=_make_label(key),
        )

    ax.set_title(f"sample={sample_id} | eps={eps}")
    ax.set_xlabel("margin loss")
    ax.set_ylabel("saliency loss")
    ax.grid(alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_best_candidate_curves_for_eps(
    eps: int,
    runs_for_eps: Dict[str, RunMeta],
    output_base: Path,
    objective_label: str,
    output_name: str,
) -> int:
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        print("[WARN] matplotlib not installed, skip plotting")
        return 0

    color_map = {
        "ga_1.0_0.0": "tab:red",
        "ga_0.5_0.5": "tab:blue",
        "ga_0.0_1.0": "tab:green",
        "nsgaii": "tab:orange",
    }

    key_to_sample_history: Dict[str, Dict[str, Path]] = {}
    for key, meta in runs_for_eps.items():
        key_to_sample_history[key] = _collect_run_histories(meta.run_dir)

    all_sets = [set(v.keys()) for v in key_to_sample_history.values() if v]
    if not all_sets:
        print(f"[WARN] Skip curve eps={eps}: no history files")
        return 0

    common_samples = sorted(set.intersection(*all_sets))
    if not common_samples:
        print(f"[WARN] Skip curve eps={eps}: no common samples among runs")
        return 0

    key_to_objective_curves: Dict[str, List[List[float]]] = {k: [] for k in key_to_sample_history.keys()}
    key_to_saliency_curves: Dict[str, List[List[float]]] = {k: [] for k in key_to_sample_history.keys()}

    for sample_id in common_samples:
        for key, sample_map in key_to_sample_history.items():
            history_path = sample_map.get(sample_id)
            if history_path is None:
                continue
            history = _read_history(history_path)
            if history is None:
                continue
            objective, saliency = history
            key_to_objective_curves[key].append(objective)
            key_to_saliency_curves[key].append(saliency)

    key_to_objective_mean: Dict[str, List[float]] = {}
    key_to_saliency_mean: Dict[str, List[float]] = {}
    for key in key_to_sample_history.keys():
        obj_mean = _curve_mean_with_last_padding(key_to_objective_curves[key])
        sal_mean = _curve_mean_with_last_padding(key_to_saliency_curves[key])
        if not obj_mean or not sal_mean:
            continue
        key_to_objective_mean[key] = obj_mean
        key_to_saliency_mean[key] = sal_mean

    if len(key_to_objective_mean) < 2:
        print(f"[WARN] Skip curve eps={eps}: not enough valid runs")
        return 0

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for key in sorted(key_to_objective_mean.keys()):
        obj_curve = key_to_objective_mean[key]
        sal_curve = key_to_saliency_mean[key]
        color = color_map.get(key)
        label = _make_label(key)

        axes[0].plot(range(1, len(obj_curve) + 1), obj_curve, linewidth=2.0, color=color, label=label)
        axes[1].plot(range(1, len(sal_curve) + 1), sal_curve, linewidth=2.0, color=color, label=label)

    axes[0].set_title(objective_label)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Mean best-candidate objective")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Saliency loss")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Mean best-candidate saliency")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"Best-candidate curves | eps={eps} | common_samples={len(common_samples)}")
    fig.tight_layout()

    out_path = output_base / f"eps_{eps}" / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return 1


def _build_runs_for_eps(
    eps: int,
    ga_runs: Dict[Tuple[int, str], RunMeta],
    nsgaii_runs: Dict[Tuple[int, str], RunMeta],
    weight_pairs: List[Tuple[float, float]],
) -> Dict[str, RunMeta]:
    runs_for_eps: Dict[str, RunMeta] = {}
    for wm, ws in weight_pairs:
        key = (eps, f"ga_{wm}_{ws}")
        run = ga_runs.get(key)
        if run is not None:
            runs_for_eps[key[1]] = run

    nsgaii_run = nsgaii_runs.get((eps, "nsgaii"))
    if nsgaii_run is not None:
        runs_for_eps["nsgaii"] = nsgaii_run

    return runs_for_eps


def _fill_ga_01_from_fallback(
    eps_list: List[int],
    primary_ga_runs: Dict[Tuple[int, str], RunMeta],
    fallback_ga_runs: Dict[Tuple[int, str], RunMeta],
) -> None:
    ga_01_key_name = "ga_0.0_1.0"
    for eps in eps_list:
        key = (eps, ga_01_key_name)
        if key not in primary_ga_runs and key in fallback_ga_runs:
            primary_ga_runs[key] = fallback_ga_runs[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare per-sample Pareto fronts between GA (3 weight pairs) and NSGAII"
    )
    parser.add_argument("--model-name", type=str, required=True, help="Model folder name, e.g. resnet18")
    parser.add_argument("--seed", type=int, default=22520691, help="Seed folder")
    parser.add_argument(
        "--ga-root",
        type=str,
        default="offical/server_run_seed/GA",
        help="GA root path up to algorithm level",
    )
    parser.add_argument(
        "--nsgaii-root",
        type=str,
        default="offical/server_run_seed/NSGAII",
        help="NSGAII root path up to algorithm level",
    )
    parser.add_argument("--strategy", type=str, default="uniform", help="Filter strategy")
    parser.add_argument("--explain-method", type=str, default="simple_gradient", help="Filter explain method")
    parser.add_argument(
        "--eps-list",
        type=str,
        default="100,50,20",
        help="Comma-separated eps values",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="margin_loss",
        choices=["margin_loss", "negative_cross_entropy_saliency", "all"],
        help="Loss type filter",
    )
    parser.add_argument(
        "--ga-weight-pairs",
        type=str,
        default="1:0,0.5:0.5,0:1",
        help="GA weight pairs wm:ws list",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="evaluate_script/pareto_compare",
        help="Output root folder",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    eps_list = [int(float(x.strip())) for x in args.eps_list.split(",") if x.strip()]
    if not eps_list:
        raise ValueError("eps-list is empty")

    weight_pairs = _parse_weight_pairs(args.ga_weight_pairs)

    ga_root = Path(args.ga_root) / str(args.seed) / args.model_name
    nsgaii_root = Path(args.nsgaii_root) / str(args.seed) / args.model_name

    ga_runs = _find_matching_runs(
        root=ga_root,
        expected_algorithm="ga",
        strategy=args.strategy,
        explain_method=args.explain_method,
        eps_list=eps_list,
        loss_type=args.loss_type,
        weight_pairs=weight_pairs,
    )
    nsgaii_runs = _find_matching_runs(
        root=nsgaii_root,
        expected_algorithm="nsgaii",
        strategy=args.strategy,
        explain_method=args.explain_method,
        eps_list=eps_list,
        loss_type=args.loss_type,
        weight_pairs=None,
    )

    ga_runs_margin = _find_matching_runs(
        root=ga_root,
        expected_algorithm="ga",
        strategy=args.strategy,
        explain_method=args.explain_method,
        eps_list=eps_list,
        loss_type="margin_loss",
        weight_pairs=weight_pairs,
    )
    nsgaii_runs_margin = _find_matching_runs(
        root=nsgaii_root,
        expected_algorithm="nsgaii",
        strategy=args.strategy,
        explain_method=args.explain_method,
        eps_list=eps_list,
        loss_type="margin_loss",
        weight_pairs=None,
    )

    ga_runs_loglikelihood = _find_matching_runs(
        root=ga_root,
        expected_algorithm="ga",
        strategy=args.strategy,
        explain_method=args.explain_method,
        eps_list=eps_list,
        loss_type="negative_cross_entropy_saliency",
        weight_pairs=weight_pairs,
    )

    # GA wm=0,ws=1 may be executed once (margin only), but should appear in both figures.
    _fill_ga_01_from_fallback(
        eps_list=eps_list,
        primary_ga_runs=ga_runs_loglikelihood,
        fallback_ga_runs=ga_runs_margin,
    )

    nsgaii_runs_loglikelihood = _find_matching_runs(
        root=nsgaii_root,
        expected_algorithm="nsgaii",
        strategy=args.strategy,
        explain_method=args.explain_method,
        eps_list=eps_list,
        loss_type="negative_cross_entropy_saliency",
        weight_pairs=None,
    )

    output_base = Path(args.output_root) / f"{args.model_name}_{args.seed}"
    output_base.mkdir(parents=True, exist_ok=True)

    total_plots = 0
    total_curve_plots = 0
    for eps in eps_list:
        runs_for_eps = _build_runs_for_eps(
            eps=eps,
            ga_runs=ga_runs,
            nsgaii_runs=nsgaii_runs,
            weight_pairs=weight_pairs,
        )
        print(runs_for_eps)

        if len(runs_for_eps) < 2:
            print(f"[WARN] Skip eps={eps}: not enough matched runs")
            continue

        key_to_sample_front: Dict[str, Dict[str, Path]] = {}
        for key, meta in runs_for_eps.items():
            key_to_sample_front[key] = _collect_run_fronts(meta.run_dir)

        all_sets = [set(v.keys()) for v in key_to_sample_front.values() if v]
        if not all_sets:
            print(f"[WARN] Skip eps={eps}: no sample front files")
            continue

        common_samples = sorted(set.intersection(*all_sets))
        if not common_samples:
            print(f"[WARN] Skip eps={eps}: no common samples among selected runs")
            continue

        for sample_id in common_samples:
            key_to_points: Dict[str, List[Tuple[float, float]]] = {}
            for key, sample_map in key_to_sample_front.items():
                points = _read_front(sample_map[sample_id])
                if points is None:
                    continue
                key_to_points[key] = points

            if len(key_to_points) < 2:
                continue

            out_path = output_base / f"eps_{eps}" / f"{sample_id.replace('/', '__')}__pareto.png"
            _plot_one_sample(sample_id=sample_id, eps=eps, key_to_points=key_to_points, output_path=out_path)
            total_plots += 1

        runs_margin = _build_runs_for_eps(
            eps=eps,
            ga_runs=ga_runs_margin,
            nsgaii_runs=nsgaii_runs_margin,
            weight_pairs=weight_pairs,
        )
        if len(runs_margin) >= 2:
            total_curve_plots += _plot_best_candidate_curves_for_eps(
                eps=eps,
                runs_for_eps=runs_margin,
                output_base=output_base,
                objective_label="Margin loss",
                output_name="best_candidate_curve__margin_vs_saliency.png",
            )

        runs_loglikelihood = _build_runs_for_eps(
            eps=eps,
            ga_runs=ga_runs_loglikelihood,
            nsgaii_runs=nsgaii_runs_loglikelihood,
            weight_pairs=weight_pairs,
        )
        if len(runs_loglikelihood) >= 2:
            total_curve_plots += _plot_best_candidate_curves_for_eps(
                eps=eps,
                runs_for_eps=runs_loglikelihood,
                output_base=output_base,
                objective_label="Negative log-likelihood",
                output_name="best_candidate_curve__loglikelihood_vs_saliency.png",
            )

    print(f"GA root: {ga_root}")
    print(f"NSGAII root: {nsgaii_root}")
    print(f"Output: {output_base}")
    print(f"Generated plots: {total_plots}")
    print(f"Generated curve figures: {total_curve_plots}")


if __name__ == "__main__":
    main()
