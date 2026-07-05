import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


@dataclass
class RunStats:
    run_dir: Path
    model: str
    strategy: str
    eps: int
    explain_method: str
    algorithm: str
    loss_type: str
    asr: float
    spearman: float
    spearman_failed_samples: int
    asr_curve: List[float]
    saliency_curve: List[float]


def _rankdata_average_ties(values: np.ndarray) -> np.ndarray:
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


def _spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("Spearman inputs must have the same number of elements")
    if x.size < 2:
        return float("nan")

    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = math.sqrt(float(np.dot(rx, rx)) * float(np.dot(ry, ry)))
    if denom < 1e-12:
        return float("nan")

    return float(np.dot(rx, ry) / denom)


def _load_gray_image_array(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    return np.asarray(image, dtype=np.float64)


def _safe_mean(values: List[float]) -> float:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def _resolve_output_dir_candidates(result: Dict[str, object], report_path: Path) -> List[Path]:
    report_parent = report_path.parent
    raw_output_dir = result.get("output_dir", "")

    candidates: List[Path] = []
    if raw_output_dir:
        path = Path(str(raw_output_dir))
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(Path.cwd() / path)
            candidates.append(report_parent / path)
            repo_root = Path(__file__).resolve().parent.parent
            candidates.append(repo_root / path)

            if path.parts:
                first = path.parts[0]
                for ancestor in [report_parent] + list(report_parent.parents):
                    if ancestor.name == first:
                        candidates.append(ancestor.parent / path)

            if len(path.parts) >= 2:
                candidates.append(report_parent / path.parts[-2] / path.parts[-1])

    class_name = result.get("class")
    image_ref = (
        result.get("resolved_image")
        or result.get("input_raw")
        or result.get("image")
        or result.get("img")
    )
    if class_name and image_ref:
        image_stem = Path(str(image_ref)).stem
        if image_stem:
            candidates.append(report_parent / str(class_name) / image_stem)

    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _compute_spearman_for_sample(result: Dict[str, object], report_path: Path) -> Tuple[Optional[float], Optional[Dict[str, object]]]:
    candidates = _resolve_output_dir_candidates(result=result, report_path=report_path)
    tried_pairs: List[Dict[str, str]] = []

    clean_map_path = None
    adv_map_path = None
    selected_output_dir = None

    for output_dir in candidates:
        clean_candidate = output_dir / "clean_map.png"
        adv_candidate = output_dir / "adv_map.png"
        tried_pairs.append(
            {
                "clean_map_path": str(clean_candidate),
                "adv_map_path": str(adv_candidate),
            }
        )
        if clean_candidate.exists() and adv_candidate.exists():
            selected_output_dir = output_dir
            clean_map_path = clean_candidate
            adv_map_path = adv_candidate
            break

    if clean_map_path is None or adv_map_path is None:
        raw_output_dir = Path(str(result.get("output_dir", "")))
        return None, {
            "reason": "missing_clean_or_adv_map",
            "output_dir": str(raw_output_dir),
            "clean_map_path": str(raw_output_dir / "clean_map.png"),
            "adv_map_path": str(raw_output_dir / "adv_map.png"),
            "tried_paths": tried_pairs,
        }

    clean_map = _load_gray_image_array(clean_map_path)
    adv_map = _load_gray_image_array(adv_map_path)

    corr = _spearman_rank_corr(clean_map, adv_map)
    if corr is None or math.isnan(corr):
        return corr, {
            "reason": "nan_spearman",
            "resolved_output_dir": str(selected_output_dir),
            "clean_map_path": str(clean_map_path),
            "adv_map_path": str(adv_map_path),
        }

    return corr, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate run statistics: ASR curve, saliency-loss curve, and LaTeX rows "
            "for margin vs negative cross entropy and saliency-guided vs uniform."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default="server_run/server_run",
        help="Root folder that contains model folders and run folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluate_script/stats_outputs",
        help="Directory to save JSON summaries and optional plots",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="weighted_sum_ga",
        help="Keep only this algorithm (weighted_sum_ga or nsgaii). Use 'all' to keep all.",
    )
    parser.add_argument(
        "--explain-method",
        type=str,
        default="simple_gradient",
        help="Keep only this explain method. Use 'all' to keep all.",
    )
    parser.add_argument(
        "--print-latex",
        action="store_true",
        help="Print compact LaTeX rows for the table in latex_table.txt",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Generate PNG plots for ASR and saliency curves",
    )
    return parser.parse_args()


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
    }
    if not approach:
        return fields

    for token in str(approach).split("__"):
        if token.startswith("strategy-"):
            fields["strategy"] = token[len("strategy-") :]
        elif token.startswith("eps-"):
            raw_eps = token[len("eps-") :]
            if re.fullmatch(r"\d+", raw_eps):
                fields["eps"] = int(raw_eps)
        elif token.startswith("exp-"):
            fields["explain_method"] = token[len("exp-") :]
        elif token.startswith("algo-"):
            fields["algorithm"] = _normalize_algorithm(token[len("algo-") :])
        elif token.startswith("fit-negative_cross_entropy_saliency"):
            fields["loss_type"] = "negative_cross_entropy_saliency"

    return fields


def _curve_mean_with_last_padding(histories: List[List[float]]) -> List[float]:
    valid = [h for h in histories if h]
    if not valid:
        return []

    max_len = max(len(h) for h in valid)
    means: List[float] = []
    for i in range(max_len):
        values = []
        for history in valid:
            if i < len(history):
                values.append(history[i])
            else:
                values.append(history[-1])
        means.append(sum(values) / len(values))
    return means


def _build_asr_curve(total_samples: int, first_success_iters: List[Optional[int]], max_iter: int) -> List[float]:
    if total_samples <= 0 or max_iter <= 0:
        return []

    curve: List[float] = []
    for iteration in range(1, max_iter + 1):
        success_count = 0
        for first_iter in first_success_iters:
            if first_iter is not None and first_iter <= iteration:
                success_count += 1
        curve.append(success_count / total_samples)
    return curve


def _extract_run_stats(run_dir: Path, model_name: str) -> Optional[RunStats]:
    report_path = run_dir / "batch_report.json"
    if not report_path.exists():
        return None

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    approach = str(report.get("approach", ""))
    meta = _parse_approach(approach)
    strategy = str(meta["strategy"])
    eps = meta["eps"]
    explain_method = str(meta["explain_method"])
    algorithm = str(meta["algorithm"])
    loss_type = str(meta["loss_type"])
    if eps is None:
        return None

    results = [r for r in report.get("results", []) if r.get("status") == "ok"]
    if not results:
        return None

    first_success_iters: List[Optional[int]] = []
    saliency_histories: List[List[float]] = []
    spearman_scores: List[float] = []
    spearman_failed_samples = 0
    max_iter = 0
    final_success_count = 0

    for result in tqdm(
        results,
        desc=f"samples {model_name}/{run_dir.name}",
        unit="img",
        leave=False,
    ):
        true_label = result.get("true_label", result.get("clean_pred", -1))
        adv_pred = result.get("adv_pred", -1)
        final_success = int(adv_pred) != int(true_label) and int(true_label) >= 0 and int(adv_pred) >= 0
        if final_success:
            final_success_count += 1

        first_iter_raw = result.get("first_success_iteration")
        first_iter = None
        if first_iter_raw is not None:
            try:
                first_iter = int(first_iter_raw)
            except (TypeError, ValueError):
                first_iter = None

        history_sal = result.get("history_saliency", [])
        if isinstance(history_sal, list):
            sal_curve = [_safe_float(v) for v in history_sal]
        else:
            sal_curve = []
        saliency_histories.append(sal_curve)
        if len(sal_curve) > max_iter:
            max_iter = len(sal_curve)

        if first_iter is None and final_success:
            first_iter = len(sal_curve) if sal_curve else 1
        first_success_iters.append(first_iter)

        corr, err = _compute_spearman_for_sample(result=result, report_path=report_path)
        if err is not None:
            spearman_failed_samples += 1
        if corr is not None:
            spearman_scores.append(corr)

    spearman = _safe_mean(spearman_scores)

    asr = final_success_count / len(results)
    asr_curve = _build_asr_curve(
        total_samples=len(results),
        first_success_iters=first_success_iters,
        max_iter=max_iter,
    )
    saliency_curve = _curve_mean_with_last_padding(saliency_histories)

    return RunStats(
        run_dir=run_dir,
        model=model_name,
        strategy=strategy,
        eps=int(eps),
        explain_method=explain_method,
        algorithm=algorithm,
        loss_type=loss_type,
        asr=asr,
        spearman=spearman,
        spearman_failed_samples=spearman_failed_samples,
        asr_curve=asr_curve,
        saliency_curve=saliency_curve,
    )


def _load_all_runs(root_dir: Path) -> List[RunStats]:
    all_runs: List[RunStats] = []
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"root dir not found: {root_dir}")

    model_dirs = [d for d in sorted(root_dir.iterdir()) if d.is_dir()]
    for model_dir in tqdm(model_dirs, desc="models", unit="model"):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        run_dirs = [d for d in sorted(model_dir.iterdir()) if d.is_dir()]
        for run_dir in tqdm(run_dirs, desc=f"runs {model_name}", unit="run", leave=False):
            if not run_dir.is_dir():
                continue
            stats = _extract_run_stats(run_dir=run_dir, model_name=model_name)
            if stats is not None:
                all_runs.append(stats)

    return all_runs


def _mean_curves(curves: List[List[float]]) -> List[float]:
    return _curve_mean_with_last_padding(curves)


def _group_runs(runs: List[RunStats], key_fn) -> Dict[Tuple[object, ...], List[RunStats]]:
    grouped: Dict[Tuple[object, ...], List[RunStats]] = {}
    for run in runs:
        key = key_fn(run)
        grouped.setdefault(key, []).append(run)
    return grouped


def _choose_best_by_asr_then_spearman(runs: List[RunStats]) -> Optional[RunStats]:
    if not runs:
        return None
    return sorted(
        runs,
        key=lambda r: (
            -_safe_float(r.asr, -1.0),
            _safe_float(r.spearman, 999.0),
            str(r.run_dir),
        ),
    )[0]


def _build_latex_rows(runs: List[RunStats]) -> List[str]:
    rows: List[str] = []
    grouped = _group_runs(
        runs,
        key_fn=lambda r: (r.model, r.eps, r.strategy, r.explain_method, r.algorithm),
    )

    for key in sorted(grouped.keys(), key=lambda x: (str(x[0]), int(x[1]), str(x[2]), str(x[3]), str(x[4]))):
        model, eps, strategy, _, _ = key
        group = grouped[key]

        margin = _choose_best_by_asr_then_spearman([g for g in group if g.loss_type == "margin_loss"])
        ce = _choose_best_by_asr_then_spearman(
            [g for g in group if g.loss_type == "negative_cross_entropy_saliency"]
        )
        if margin is None or ce is None:
            continue

        strategy_text = "Saliency-guided" if strategy == "saliency_guided" else "Uniform"
        row = (
            f"& {eps} & {strategy_text} "
            f"& {margin.asr * 100.0:.2f} & {margin.spearman:.4f} "
            f"& {ce.asr * 100.0:.2f} & {ce.spearman:.4f} \\\\"
        )
        rows.append((str(model), row))

    rendered: List[str] = []
    current_model = None
    for model, row in rows:
        if current_model != model:
            rendered.append(f"% Model: {model}")
            current_model = model
        rendered.append(row)
    return rendered


def _build_pair_curves_loss(runs: List[RunStats]) -> List[Dict[str, object]]:
    grouped = _group_runs(
        runs,
        key_fn=lambda r: (r.model, r.eps, r.strategy, r.explain_method, r.algorithm),
    )
    output: List[Dict[str, object]] = []

    for key, group in grouped.items():
        model, eps, strategy, explain_method, algorithm = key
        margin_runs = [g for g in group if g.loss_type == "margin_loss"]
        ce_runs = [g for g in group if g.loss_type == "negative_cross_entropy_saliency"]
        if not margin_runs or not ce_runs:
            continue

        output.append(
            {
                "model": model,
                "eps": eps,
                "strategy": strategy,
                "explain_method": explain_method,
                "algorithm": algorithm,
                "margin_loss": {
                    "asr_curve": _mean_curves([r.asr_curve for r in margin_runs]),
                    "saliency_curve": _mean_curves([r.saliency_curve for r in margin_runs]),
                    "final_asr": sum(r.asr for r in margin_runs) / len(margin_runs),
                },
                "negative_cross_entropy_saliency": {
                    "asr_curve": _mean_curves([r.asr_curve for r in ce_runs]),
                    "saliency_curve": _mean_curves([r.saliency_curve for r in ce_runs]),
                    "final_asr": sum(r.asr for r in ce_runs) / len(ce_runs),
                },
            }
        )

    return output


def _build_pair_curves_init(runs: List[RunStats]) -> List[Dict[str, object]]:
    grouped = _group_runs(
        runs,
        key_fn=lambda r: (r.model, r.eps, r.loss_type, r.explain_method, r.algorithm),
    )
    output: List[Dict[str, object]] = []

    for key, group in grouped.items():
        model, eps, loss_type, explain_method, algorithm = key
        saliency_runs = [g for g in group if g.strategy == "saliency_guided"]
        uniform_runs = [g for g in group if g.strategy == "uniform"]
        if not saliency_runs or not uniform_runs:
            continue

        output.append(
            {
                "model": model,
                "eps": eps,
                "loss_type": loss_type,
                "explain_method": explain_method,
                "algorithm": algorithm,
                "saliency_guided": {
                    "asr_curve": _mean_curves([r.asr_curve for r in saliency_runs]),
                    "saliency_curve": _mean_curves([r.saliency_curve for r in saliency_runs]),
                    "final_asr": sum(r.asr for r in saliency_runs) / len(saliency_runs),
                },
                "uniform": {
                    "asr_curve": _mean_curves([r.asr_curve for r in uniform_runs]),
                    "saliency_curve": _mean_curves([r.saliency_curve for r in uniform_runs]),
                    "final_asr": sum(r.asr for r in uniform_runs) / len(uniform_runs),
                },
            }
        )

    return output


def _build_overall_compare_loss(loss_pairs: List[Dict[str, object]]) -> Dict[str, object]:
    margin_asr_curves = [item["margin_loss"]["asr_curve"] for item in loss_pairs]
    ce_asr_curves = [item["negative_cross_entropy_saliency"]["asr_curve"] for item in loss_pairs]
    margin_sal_curves = [item["margin_loss"]["saliency_curve"] for item in loss_pairs]
    ce_sal_curves = [item["negative_cross_entropy_saliency"]["saliency_curve"] for item in loss_pairs]

    margin_final_asr = [float(item["margin_loss"]["final_asr"]) for item in loss_pairs]
    ce_final_asr = [float(item["negative_cross_entropy_saliency"]["final_asr"]) for item in loss_pairs]

    return {
        "num_pairs": len(loss_pairs),
        "averaged_over": "all_models_all_eps_all_strategies_after_filters",
        "margin_loss": {
            "asr_curve": _mean_curves(margin_asr_curves),
            "saliency_curve": _mean_curves(margin_sal_curves),
            "final_asr": _safe_mean(margin_final_asr),
        },
        "negative_cross_entropy_saliency": {
            "asr_curve": _mean_curves(ce_asr_curves),
            "saliency_curve": _mean_curves(ce_sal_curves),
            "final_asr": _safe_mean(ce_final_asr),
        },
    }


def _build_overall_compare_init(init_pairs: List[Dict[str, object]]) -> Dict[str, object]:
    sal_asr_curves = [item["saliency_guided"]["asr_curve"] for item in init_pairs]
    uni_asr_curves = [item["uniform"]["asr_curve"] for item in init_pairs]
    sal_sal_curves = [item["saliency_guided"]["saliency_curve"] for item in init_pairs]
    uni_sal_curves = [item["uniform"]["saliency_curve"] for item in init_pairs]

    sal_final_asr = [float(item["saliency_guided"]["final_asr"]) for item in init_pairs]
    uni_final_asr = [float(item["uniform"]["final_asr"]) for item in init_pairs]

    return {
        "num_pairs": len(init_pairs),
        "averaged_over": "all_models_all_eps_all_loss_types_after_filters",
        "saliency_guided": {
            "asr_curve": _mean_curves(sal_asr_curves),
            "saliency_curve": _mean_curves(sal_sal_curves),
            "final_asr": _safe_mean(sal_final_asr),
        },
        "uniform": {
            "asr_curve": _mean_curves(uni_asr_curves),
            "saliency_curve": _mean_curves(uni_sal_curves),
            "final_asr": _safe_mean(uni_final_asr),
        },
    }


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _plot_pair_curves_loss(loss_pairs: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        print("[WARN] matplotlib not installed, skip plotting")
        return

    plot_dir = output_dir / "plots" / "compare_loss"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for item in loss_pairs:
        title = (
            f"{item['model']} | eps={item['eps']} | {item['strategy']} | "
            f"{item['explain_method']} | {item['algorithm']}"
        )

        margin_asr = item["margin_loss"]["asr_curve"]
        ce_asr = item["negative_cross_entropy_saliency"]["asr_curve"]
        margin_sal = item["margin_loss"]["saliency_curve"]
        ce_sal = item["negative_cross_entropy_saliency"]["saliency_curve"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

        x1 = list(range(1, len(margin_asr) + 1))
        x2 = list(range(1, len(ce_asr) + 1))
        axes[0].plot(x1, margin_asr, label="margin_loss", linewidth=2)
        axes[0].plot(x2, ce_asr, label="negative_cross_entropy", linewidth=2)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("ASR")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        x3 = list(range(1, len(margin_sal) + 1))
        x4 = list(range(1, len(ce_sal) + 1))
        axes[1].plot(x3, margin_sal, label="margin_loss", linewidth=2)
        axes[1].plot(x4, ce_sal, label="negative_cross_entropy", linewidth=2)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Mean saliency loss")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.suptitle(title)
        fig.tight_layout()

        file_name = (
            f"{item['model']}__eps-{item['eps']}__strategy-{item['strategy']}"
            f"__exp-{item['explain_method']}__algo-{item['algorithm']}.png"
        )
        fig.savefig(plot_dir / file_name, dpi=150)
        plt.close(fig)


def _plot_pair_curves_init(init_pairs: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        return

    plot_dir = output_dir / "plots" / "compare_init"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for item in init_pairs:
        title = (
            f"{item['model']} | eps={item['eps']} | {item['loss_type']} | "
            f"{item['explain_method']} | {item['algorithm']}"
        )

        sal_asr = item["saliency_guided"]["asr_curve"]
        uni_asr = item["uniform"]["asr_curve"]
        sal_curve = item["saliency_guided"]["saliency_curve"]
        uni_curve = item["uniform"]["saliency_curve"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

        axes[0].plot(range(1, len(sal_asr) + 1), sal_asr, label="saliency_guided", linewidth=2)
        axes[0].plot(range(1, len(uni_asr) + 1), uni_asr, label="uniform", linewidth=2)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("ASR")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(range(1, len(sal_curve) + 1), sal_curve, label="saliency_guided", linewidth=2)
        axes[1].plot(range(1, len(uni_curve) + 1), uni_curve, label="uniform", linewidth=2)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Mean saliency loss")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.suptitle(title)
        fig.tight_layout()

        file_name = (
            f"{item['model']}__eps-{item['eps']}__loss-{item['loss_type']}"
            f"__exp-{item['explain_method']}__algo-{item['algorithm']}.png"
        )
        fig.savefig(plot_dir / file_name, dpi=150)
        plt.close(fig)


def _plot_overall_compare_loss(overall: Dict[str, object], output_dir: Path) -> None:
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        return

    plot_dir = output_dir / "plots" / "compare_loss"
    plot_dir.mkdir(parents=True, exist_ok=True)

    margin_asr = overall["margin_loss"]["asr_curve"]
    ce_asr = overall["negative_cross_entropy_saliency"]["asr_curve"]
    margin_sal = overall["margin_loss"]["saliency_curve"]
    ce_sal = overall["negative_cross_entropy_saliency"]["saliency_curve"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(range(1, len(margin_asr) + 1), margin_asr, label="margin_loss", linewidth=2.2)
    axes[0].plot(range(1, len(ce_asr) + 1), ce_asr, label="negative_cross_entropy", linewidth=2.2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("ASR")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(range(1, len(margin_sal) + 1), margin_sal, label="margin_loss", linewidth=2.2)
    axes[1].plot(range(1, len(ce_sal) + 1), ce_sal, label="negative_cross_entropy", linewidth=2.2)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Mean saliency loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"compare_loss overall average (pairs={overall['num_pairs']})")
    fig.tight_layout()
    fig.savefig(plot_dir / "overall__compare_loss.png", dpi=170)
    plt.close(fig)


def _plot_overall_compare_init(overall: Dict[str, object], output_dir: Path) -> None:
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        return

    plot_dir = output_dir / "plots" / "compare_init"
    plot_dir.mkdir(parents=True, exist_ok=True)

    sal_asr = overall["saliency_guided"]["asr_curve"]
    uni_asr = overall["uniform"]["asr_curve"]
    sal_curve = overall["saliency_guided"]["saliency_curve"]
    uni_curve = overall["uniform"]["saliency_curve"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(range(1, len(sal_asr) + 1), sal_asr, label="saliency_guided", linewidth=2.2)
    axes[0].plot(range(1, len(uni_asr) + 1), uni_asr, label="uniform", linewidth=2.2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("ASR")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(range(1, len(sal_curve) + 1), sal_curve, label="saliency_guided", linewidth=2.2)
    axes[1].plot(range(1, len(uni_curve) + 1), uni_curve, label="uniform", linewidth=2.2)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Mean saliency loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"compare_init overall average (pairs={overall['num_pairs']})")
    fig.tight_layout()
    fig.savefig(plot_dir / "overall__compare_init.png", dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    root_dir = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = _load_all_runs(root_dir)
    if not all_runs:
        raise ValueError(f"No valid run found under: {root_dir}")

    filtered = all_runs
    if args.algorithm.lower() != "all":
        target_algo = _normalize_algorithm(args.algorithm)
        filtered = [r for r in filtered if r.algorithm == target_algo]

    if args.explain_method.lower() != "all":
        filtered = [r for r in filtered if r.explain_method == args.explain_method]

    if not filtered:
        raise ValueError("No run left after filtering")

    loss_pairs = _build_pair_curves_loss(filtered)
    init_pairs = _build_pair_curves_init(filtered)
    overall_compare_loss = _build_overall_compare_loss(loss_pairs)
    overall_compare_init = _build_overall_compare_init(init_pairs)
    latex_rows = _build_latex_rows(filtered)

    _save_json(output_dir / "all_runs.json", [r.__dict__ | {"run_dir": str(r.run_dir)} for r in filtered])
    _save_json(output_dir / "compare_loss_curves.json", loss_pairs)
    _save_json(output_dir / "compare_init_curves.json", init_pairs)
    _save_json(output_dir / "compare_loss_curves_overall.json", overall_compare_loss)
    _save_json(output_dir / "compare_init_curves_overall.json", overall_compare_init)
    _save_json(output_dir / "latex_rows.json", latex_rows)

    if args.make_plots:
        _plot_pair_curves_loss(loss_pairs, output_dir)
        _plot_pair_curves_init(init_pairs, output_dir)
        _plot_overall_compare_loss(overall_compare_loss, output_dir)
        _plot_overall_compare_init(overall_compare_init, output_dir)

    print(f"Loaded runs: {len(all_runs)}")
    print(f"Runs after filter: {len(filtered)}")
    print(f"Loss pairs: {len(loss_pairs)}")
    print(f"Init pairs: {len(init_pairs)}")
    print(f"Overall compare_loss pairs: {overall_compare_loss['num_pairs']}")
    print(f"Overall compare_init pairs: {overall_compare_init['num_pairs']}")
    total_spearman_failed = sum(r.spearman_failed_samples for r in filtered)
    print(f"Spearman failed samples: {total_spearman_failed}")
    print(f"Output dir: {output_dir}")

    if args.print_latex:
        print("\n=== LaTeX Rows ===")
        for row in latex_rows:
            print(row)


if __name__ == "__main__":
    main()




