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

    output_base = Path(args.output_root) / f"{args.model_name}_{args.seed}"
    output_base.mkdir(parents=True, exist_ok=True)

    total_plots = 0
    for eps in eps_list:
        runs_for_eps: Dict[str, RunMeta] = {}
        for wm, ws in weight_pairs:
            key = (eps, f"ga_{wm}_{ws}")
            run = ga_runs.get(key)
            if run is not None:
                runs_for_eps[key[1]] = run

        nsgaii_run = nsgaii_runs.get((eps, "nsgaii"))
        if nsgaii_run is not None:
            runs_for_eps["nsgaii"] = nsgaii_run

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

    print(f"GA root: {ga_root}")
    print(f"NSGAII root: {nsgaii_root}")
    print(f"Output: {output_base}")
    print(f"Generated plots: {total_plots}")


if __name__ == "__main__":
    main()
