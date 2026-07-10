import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_algorithm(name: str) -> str:
    lower = str(name or "").strip().lower()
    if lower in {"", "default", "weighted_sum", "weighted_sum_ga", "wga"}:
        return "weighted_sum_ga"
    return lower


def _parse_approach(approach: str) -> Dict[str, object]:
    fields: Dict[str, object] = {
        "strategy": "unknown",
        "eps": None,
        "explain_method": "unknown",
        "algorithm": "weighted_sum_ga",
        "loss_type": "margin_loss",
        "w_m": None,
        "w_s": None,
    }
    if not approach:
        return fields

    for token in str(approach).split("__"):
        if token.startswith("strategy-"):
            fields["strategy"] = token[len("strategy-") :]
        elif token.startswith("eps-"):
            raw_eps = token[len("eps-") :]
            try:
                eps_value = float(raw_eps)
                fields["eps"] = int(eps_value) if eps_value.is_integer() else eps_value
            except ValueError:
                pass
        elif token.startswith("exp-"):
            fields["explain_method"] = token[len("exp-") :]
        elif token.startswith("algo-"):
            fields["algorithm"] = _normalize_algorithm(token[len("algo-") :])
        elif token.startswith("wm-"):
            fields["w_m"] = _safe_float(token[len("wm-") :], default=float("nan"))
        elif token.startswith("w_m-"):
            fields["w_m"] = _safe_float(token[len("w_m-") :], default=float("nan"))
        elif token.startswith("ws-"):
            fields["w_s"] = _safe_float(token[len("ws-") :], default=float("nan"))
        elif token.startswith("w_s-"):
            fields["w_s"] = _safe_float(token[len("w_s-") :], default=float("nan"))
        elif token.startswith("wc-"):
            fields["w_s"] = _safe_float(token[len("wc-") :], default=float("nan"))
        elif token.startswith("w_c-"):
            fields["w_s"] = _safe_float(token[len("w_c-") :], default=float("nan"))
        elif token.startswith("fit-negative_cross_entropy_saliency"):
            fields["loss_type"] = "negative_cross_entropy_saliency"

    if fields["w_m"] is not None and math.isnan(float(fields["w_m"])):
        fields["w_m"] = None
    if fields["w_s"] is not None and math.isnan(float(fields["w_s"])):
        fields["w_s"] = None

    return fields


def _parse_optional_float_arg(raw: str, arg_name: str) -> Optional[float]:
    text = str(raw or "").strip().lower()
    if text in {"", "all"}:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid value for {arg_name}: {raw}. Use number or 'all'.") from exc


def _parse_weight_pairs(raw: Optional[str]) -> Optional[List[Tuple[float, float]]]:
    if raw is None:
        return None

    text = str(raw).strip()
    if not text:
        return None

    pairs: List[Tuple[float, float]] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid pair '{item}'. Expected wm:ws, e.g. 1:0")
        left, right = item.split(":", 1)
        try:
            pairs.append((float(left.strip()), float(right.strip())))
        except ValueError as exc:
            raise ValueError(f"Invalid pair '{item}'. wm and ws must be numbers") from exc

    return pairs or None


def _is_close(a: Optional[float], b: float, tol: float = 1e-9) -> bool:
    if a is None:
        return False
    return abs(float(a) - float(b)) <= tol


def _has_summary_files(run_dir: Path) -> bool:
    return any(run_dir.glob("*/*/summary.json")) or any(run_dir.glob("*/*/summarize.json"))


def _load_run_results(run_dir: Path) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    summary_paths = list(run_dir.glob("*/*/summary.json")) + list(run_dir.glob("*/*/summarize.json"))
    seen = set()

    for summary_path in sorted(summary_paths):
        key = str(summary_path)
        if key in seen:
            continue
        seen.add(key)
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        if isinstance(payload, dict):
            results.append(payload)

    return results


def _extract_first_success_query(sample: Dict[str, object]) -> Optional[int]:
    first_iter_raw = sample.get("first_success_iteration")
    if first_iter_raw is None:
        return None
    try:
        first_iter = int(first_iter_raw)
    except (TypeError, ValueError):
        return None
    return first_iter if first_iter > 0 else None


def _is_attack_success(sample: Dict[str, object]) -> bool:
    true_label = sample.get("true_label", sample.get("clean_pred", -1))
    adv_pred = sample.get("adv_pred", -1)
    try:
        true_label_i = int(true_label)
        adv_pred_i = int(adv_pred)
    except (TypeError, ValueError):
        return False
    return true_label_i >= 0 and adv_pred_i >= 0 and adv_pred_i != true_label_i


def _summarize_queries(queries: List[int], bins: int) -> Dict[str, object]:
    if not queries:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p90": None,
            "histogram": {"bin_edges": [], "counts": []},
        }

    arr = np.asarray(queries, dtype=np.float64)
    hist_counts, hist_edges = np.histogram(arr, bins=max(1, bins))

    return {
        "count": int(arr.size),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "histogram": {
            "bin_edges": [float(v) for v in hist_edges.tolist()],
            "counts": [int(v) for v in hist_counts.tolist()],
        },
    }


def _scan_run_dirs(root_dir: Path) -> List[Tuple[str, Path]]:
    pairs: List[Tuple[str, Path]] = []

    if _has_summary_files(root_dir):
        pairs.append(("unknown", root_dir))
        return pairs

    direct_children = [d for d in sorted(root_dir.iterdir()) if d.is_dir()]
    direct_run_dirs = [d for d in direct_children if _has_summary_files(d)]
    if direct_run_dirs:
        for run_dir in direct_run_dirs:
            pairs.append(("unknown", run_dir))
        return pairs

    for model_dir in sorted(root_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for run_dir in sorted(model_dir.iterdir()):
            if run_dir.is_dir() and _has_summary_files(run_dir):
                pairs.append((model_name, run_dir))

    return pairs


def _scan_run_dirs_from_all_runs_json(all_runs_json: Path) -> List[Tuple[str, Path]]:
    with open(all_runs_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Invalid all_runs JSON format: expected list, got {type(payload).__name__}")

    pairs: List[Tuple[str, Path]] = []
    seen = set()
    for item in payload:
        if not isinstance(item, dict):
            continue

        run_dir_raw = item.get("run_dir")
        if not run_dir_raw:
            continue

        run_dir = Path(str(run_dir_raw))
        if not run_dir.exists() or not run_dir.is_dir():
            continue

        key = str(run_dir.resolve())
        if key in seen:
            continue
        seen.add(key)

        model_name = str(item.get("model", "unknown"))
        pairs.append((model_name, run_dir))

    return pairs


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _plot_per_run_histograms(rows: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        print("[WARN] matplotlib not installed, skip plotting")
        return

    plot_dir = output_dir / "plots" / "query_distribution"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for item in rows:
        queries = item.get("success_queries", [])
        if not queries:
            continue

        run_name = str(item.get("run_name", "run"))
        model_name = str(item.get("model", "unknown"))

        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.hist(queries, bins=min(30, max(5, len(set(queries)))), color="#1f77b4", alpha=0.85)
        ax.set_title(f"Success Query Distribution | {model_name} | {run_name}")
        ax.set_xlabel("First success query")
        ax.set_ylabel("Sample count")
        ax.grid(alpha=0.25)

        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", f"{model_name}__{run_name}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{safe_name}.png", dpi=160)
        plt.close(fig)


def _plot_grouped_boxplot(rows: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        return

    grouped: Dict[str, List[int]] = {}
    for item in rows:
        strategy = str(item.get("strategy", "unknown"))
        loss_type = str(item.get("loss_type", "unknown"))
        key = f"{strategy} | {loss_type}"
        grouped.setdefault(key, []).extend([int(v) for v in item.get("success_queries", [])])

    labels = [k for k, v in grouped.items() if v]
    if not labels:
        return

    data = [grouped[k] for k in labels]

    plot_dir = output_dir / "plots" / "query_distribution"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5.2))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_title("Success Query Distribution by Strategy/Loss")
    ax.set_xlabel("Group")
    ax.set_ylabel("First success query")
    ax.grid(alpha=0.25)
    plt.xticks(rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(plot_dir / "overall_grouped_boxplot.png", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze distribution of successful query counts (first success iteration)."
    )
    parser.add_argument("--result-root", type=str, default="server_run/server_run", help="Root folder with model/run directories")
    parser.add_argument("--run-score-output-dir", type=str, default=None, help="Output directory produced by run_score.py (contains all_runs.json)")
    parser.add_argument("--all-runs-json", type=str, default=None, help="Path to all_runs.json from run_score output")
    parser.add_argument("--output-dir", type=str, default="evaluate_script/query_distribution_outputs", help="Directory to save JSON and plots")
    parser.add_argument("--algorithm", type=str, default="all", help="Filter algorithm (weighted_sum_ga or nsgaii). Use all to keep all")
    parser.add_argument("--explain-method", type=str, default="all", help="Filter explain method. Use all to keep all")
    parser.add_argument("--w-m", type=str, default="all", help="Filter by w_m. Use numeric value or all")
    parser.add_argument("--w-c", type=str, default="all", help="Backward-compatible alias of --w-s")
    parser.add_argument("--w-s", type=str, default="all", help="Filter by w_s. Use numeric value or all")
    parser.add_argument("--weight-pairs", type=str, default=None, help="Comma-separated wm:ws pairs, e.g. 1:0,0.5:0.5")
    parser.add_argument("--bins", type=int, default=20, help="Number of bins for histogram summary JSON")
    parser.add_argument("--make-plots", action="store_true", help="Generate plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_pairs: List[Tuple[str, Path]]
    if args.all_runs_json:
        all_runs_path = Path(args.all_runs_json)
        if not all_runs_path.exists():
            raise FileNotFoundError(f"all_runs JSON not found: {all_runs_path}")
        run_pairs = _scan_run_dirs_from_all_runs_json(all_runs_path)
    elif args.run_score_output_dir:
        score_out_dir = Path(args.run_score_output_dir)
        all_runs_path = score_out_dir / "all_runs.json"
        if not all_runs_path.exists():
            raise FileNotFoundError(f"all_runs.json not found in run_score output dir: {score_out_dir}")
        run_pairs = _scan_run_dirs_from_all_runs_json(all_runs_path)
    else:
        result_root = Path(args.result_root)
        if not result_root.exists() or not result_root.is_dir():
            raise FileNotFoundError(f"result root not found: {result_root}")
        run_pairs = _scan_run_dirs(result_root)

    target_algo = None
    if args.algorithm.lower() != "all":
        target_algo = _normalize_algorithm(args.algorithm)

    target_explain_method = None
    if args.explain_method.lower() != "all":
        target_explain_method = args.explain_method

    target_w_m = _parse_optional_float_arg(args.w_m, "--w-m")
    raw_w_s = args.w_s if str(args.w_s).strip().lower() != "all" else args.w_c
    target_w_s = _parse_optional_float_arg(raw_w_s, "--w-s")
    target_pairs = _parse_weight_pairs(args.weight_pairs)

    if not run_pairs:
        raise ValueError("No run directories with summary files found from provided input source")

    rows: List[Dict[str, object]] = []

    for model_name, run_dir in run_pairs:
        approach = run_dir.name
        meta = _parse_approach(approach)

        if target_algo is not None and str(meta["algorithm"]) != target_algo:
            continue
        if target_explain_method is not None and str(meta["explain_method"]) != target_explain_method:
            continue
        if target_w_m is not None and not _is_close(meta["w_m"], target_w_m):
            continue
        if target_w_s is not None and not _is_close(meta["w_s"], target_w_s):
            continue
        if target_pairs and not any(_is_close(meta["w_m"], wm) and _is_close(meta["w_s"], ws) for wm, ws in target_pairs):
            continue

        samples = _load_run_results(run_dir)
        ok_samples = [s for s in samples if isinstance(s, dict) and s.get("status") == "ok"]

        success_queries: List[int] = []
        success_count = 0
        for sample in ok_samples:
            if not _is_attack_success(sample):
                continue
            success_count += 1
            q = _extract_first_success_query(sample)
            if q is not None:
                success_queries.append(q)

        distribution = _summarize_queries(success_queries, bins=args.bins)

        rows.append(
            {
                "model": model_name,
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "strategy": meta["strategy"],
                "eps": meta["eps"],
                "explain_method": meta["explain_method"],
                "algorithm": meta["algorithm"],
                "loss_type": meta["loss_type"],
                "w_m": meta["w_m"],
                "w_s": meta["w_s"],
                "num_samples_total": len(samples),
                "num_samples_ok": len(ok_samples),
                "num_attack_success": success_count,
                "asr": (success_count / len(ok_samples)) if ok_samples else 0.0,
                "num_success_with_query": len(success_queries),
                "success_queries": success_queries,
                "query_distribution": distribution,
            }
        )

    if not rows:
        raise ValueError("No run left after filtering")

    # Flatten global distribution across all runs.
    global_queries: List[int] = []
    for row in rows:
        global_queries.extend([int(v) for v in row.get("success_queries", [])])

    global_summary = {
        "num_runs": len(rows),
        "num_success_queries": len(global_queries),
        "query_distribution": _summarize_queries(global_queries, bins=args.bins),
    }

    _save_json(output_dir / "query_distribution_per_run.json", rows)
    _save_json(output_dir / "query_distribution_overall.json", global_summary)

    if args.make_plots:
        _plot_per_run_histograms(rows, output_dir)
        _plot_grouped_boxplot(rows, output_dir)

    print(f"Runs analyzed: {len(rows)}")
    print(f"Global success-query samples: {len(global_queries)}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
