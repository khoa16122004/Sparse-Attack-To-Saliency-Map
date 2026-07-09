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
    w_m: Optional[float]
    w_s: Optional[float]
    asr: float
    spearman: float
    spearman_failed_samples: int
    num_samples_total: int
    num_samples_ok: int
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
        "--all-runs-json",
        type=str,
        default=None,
        help=(
            "Optional path to precomputed all_runs.json. "
            "When provided, skip scanning root and reuse this file."
        ),
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
        "--w-m",
        type=str,
        default="all",
        help="Filter by w_m (margin weight). Use numeric value or 'all'.",
    )
    parser.add_argument(
        "--w-c",
        type=str,
        default="all",
        help="[Backward-compatible alias] Same as --w-s.",
    )
    parser.add_argument(
        "--w-s",
        type=str,
        default="all",
        help="Filter by ws (saliency weight). Use numeric value or 'all'.",
    )
    parser.add_argument(
        "--weight-pairs",
        type=str,
        default=None,
        help=(
            "Comma-separated wm:ws pairs, e.g. '0.5:0.5,1:0'. "
            "When set, only runs that match at least one pair are kept."
        ),
    )
    parser.add_argument(
        "--print-latex",
        action="store_true",
        help="Print compact LaTeX rows for the table in latex_table.txt",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="[Deprecated] Ignored in this script. Use evaluate_script/plot_from_all_runs.py for plots.",
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
            if re.fullmatch(r"\d+", raw_eps):
                fields["eps"] = int(raw_eps)
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
        raise ValueError(f"Invalid value for {arg_name}: {raw}. Use a number or 'all'.") from exc


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
            raise ValueError(
                f"Invalid weight pair '{item}'. Expected format wm:ws, e.g. 0.5:0.5"
            )
        left, right = item.split(":", 1)
        try:
            pairs.append((float(left.strip()), float(right.strip())))
        except ValueError as exc:
            raise ValueError(
                f"Invalid weight pair '{item}'. Both wm and ws must be numbers."
            ) from exc

    return pairs or None


def _is_close(a: Optional[float], b: float, tol: float = 1e-9) -> bool:
    if a is None:
        return False
    return abs(float(a) - float(b)) <= tol


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


def _extract_run_stats(
    run_dir: Path,
    model_name: str,
    target_algorithm: Optional[str] = None,
    target_explain_method: Optional[str] = None,
    target_w_m: Optional[float] = None,
    target_w_s: Optional[float] = None,
    target_pairs: Optional[List[Tuple[float, float]]] = None,
) -> Optional[RunStats]:
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
    w_m = meta["w_m"]
    w_s = meta["w_s"]
    if eps is None:
        return None

    # Skip runs that do not match requested filters before expensive sample loops.
    if target_algorithm is not None and algorithm != target_algorithm:
        return None
    if target_explain_method is not None and explain_method != target_explain_method:
        return None
    if target_w_m is not None and not _is_close(w_m, target_w_m):
        return None
    if target_w_s is not None and not _is_close(w_s, target_w_s):
        return None
    if target_pairs and not any(_is_close(w_m, wm) and _is_close(w_s, ws) for wm, ws in target_pairs):
        return None

    all_results = report.get("results", [])
    results = [r for r in all_results if r.get("status") == "ok"]
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
        w_m=w_m,
        w_s=w_s,
        asr=asr,
        spearman=spearman,
        spearman_failed_samples=spearman_failed_samples,
        num_samples_total=len(all_results),
        num_samples_ok=len(results),
        asr_curve=asr_curve,
        saliency_curve=saliency_curve,
    )


def _load_all_runs(
    root_dir: Path,
    target_algorithm: Optional[str] = None,
    target_explain_method: Optional[str] = None,
    target_w_m: Optional[float] = None,
    target_w_s: Optional[float] = None,
    target_pairs: Optional[List[Tuple[float, float]]] = None,
) -> List[RunStats]:
    all_runs: List[RunStats] = []
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"root dir not found: {root_dir}")

    model_dirs = [d for d in sorted(root_dir.iterdir()) if d.is_dir()]
    for model_dir in tqdm(model_dirs, desc="models", unit="model"):
        print(model_dir)
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        run_dirs = [d for d in sorted(model_dir.iterdir()) if d.is_dir()]
        for run_dir in tqdm(run_dirs, desc=f"runs {model_name}", unit="run", leave=False):
            print(run_dir)
            if not run_dir.is_dir():
                continue
            stats = _extract_run_stats(
                run_dir=run_dir,
                model_name=model_name,
                target_algorithm=target_algorithm,
                target_explain_method=target_explain_method,
                target_w_m=target_w_m,
                target_w_s=target_w_s,
                target_pairs=target_pairs,
            )
            raise
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


def _build_grouped_fourway_curves(runs: List[RunStats]) -> List[Dict[str, object]]:
    grouped = _group_runs(
        runs,
        key_fn=lambda r: (r.model, r.eps, r.explain_method, r.algorithm),
    )
    output: List[Dict[str, object]] = []

    combo_specs = [
        ("saliency_guided_negative_ce", "saliency_guided", "negative_cross_entropy_saliency"),
        ("uniform_margin", "uniform", "margin_loss"),
        ("saliency_guided_margin", "saliency_guided", "margin_loss"),
        ("uniform_negative_ce", "uniform", "negative_cross_entropy_saliency"),
    ]

    for key, group in grouped.items():
        model, eps, explain_method, algorithm = key
        combo_payload: Dict[str, object] = {}
        complete = True

        for combo_name, strategy, loss_type in combo_specs:
            selected = [r for r in group if r.strategy == strategy and r.loss_type == loss_type]
            if not selected:
                complete = False
                break

            combo_payload[combo_name] = {
                "strategy": strategy,
                "loss_type": loss_type,
                "asr_curve": _mean_curves([r.asr_curve for r in selected]),
                "saliency_curve": _mean_curves([r.saliency_curve for r in selected]),
                "final_asr": _safe_mean([r.asr for r in selected]),
            }

        if not complete:
            continue

        output.append(
            {
                "model": model,
                "eps": eps,
                "explain_method": explain_method,
                "algorithm": algorithm,
                **combo_payload,
            }
        )

    return output


def _build_overall_grouped_fourway(grouped_items: List[Dict[str, object]]) -> Dict[str, object]:
    combo_names = [
        "saliency_guided_negative_ce",
        "uniform_margin",
        "saliency_guided_margin",
        "uniform_negative_ce",
    ]

    output: Dict[str, object] = {
        "num_groups": len(grouped_items),
        "averaged_over": "all_models_all_eps_after_filters",
    }

    for combo_name in combo_names:
        asr_curves = [item[combo_name]["asr_curve"] for item in grouped_items]
        sal_curves = [item[combo_name]["saliency_curve"] for item in grouped_items]
        final_asr = [float(item[combo_name]["final_asr"]) for item in grouped_items]

        output[combo_name] = {
            "asr_curve": _mean_curves(asr_curves),
            "saliency_curve": _mean_curves(sal_curves),
            "final_asr": _safe_mean(final_asr),
        }

    return output


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _runstats_from_dict(payload: Dict[str, object]) -> RunStats:
    return RunStats(
        run_dir=Path(str(payload.get("run_dir", ""))),
        model=str(payload.get("model", "")),
        strategy=str(payload.get("strategy", "unknown")),
        eps=int(payload.get("eps", 0)),
        explain_method=str(payload.get("explain_method", "unknown")),
        algorithm=str(payload.get("algorithm", "weighted_sum_ga")),
        loss_type=str(payload.get("loss_type", "margin_loss")),
        w_m=payload.get("w_m"),
        w_s=payload.get("w_s"),
        asr=float(payload.get("asr", 0.0)),
        spearman=float(payload.get("spearman", float("nan"))),
        spearman_failed_samples=int(payload.get("spearman_failed_samples", 0)),
        num_samples_total=int(payload.get("num_samples_total", 0)),
        num_samples_ok=int(payload.get("num_samples_ok", 0)),
        asr_curve=[float(v) for v in payload.get("asr_curve", [])],
        saliency_curve=[float(v) for v in payload.get("saliency_curve", [])],
    )


def _load_runs_from_json(path: Path) -> List[RunStats]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Invalid all_runs JSON format: expected list, got {type(raw).__name__}")

    runs: List[RunStats] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        runs.append(_runstats_from_dict(item))
    return runs


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
        axes[0].plot(x2, ce_asr, label="LogLikeLihood", linewidth=2)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("ASR")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        x3 = list(range(1, len(margin_sal) + 1))
        x4 = list(range(1, len(ce_sal) + 1))
        axes[1].plot(x3, margin_sal, label="margin_loss", linewidth=2)
        axes[1].plot(x4, ce_sal, label="LogLikeLihood", linewidth=2)
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
    axes[0].plot(range(1, len(ce_asr) + 1), ce_asr, label="LogLikeLihood", linewidth=2.2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("ASR")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(range(1, len(margin_sal) + 1), margin_sal, label="margin_loss", linewidth=2.2)
    axes[1].plot(range(1, len(ce_sal) + 1), ce_sal, label="LogLikeLihood", linewidth=2.2)
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


def _plot_grouped_fourway(grouped_items: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        return

    plot_dir = output_dir / "plots" / "compare_grouped"
    plot_dir.mkdir(parents=True, exist_ok=True)

    style = {
        "saliency_guided_negative_ce": {
            "label": "Saliency Guided + LogLikeLihood",
            "color": "tab:red",
        },
        "uniform_margin": {
            "label": "Uniform + Margin",
            "color": "tab:blue",
        },
        "saliency_guided_margin": {
            "label": "Saliency Guided + Margin",
            "color": "tab:green",
        },
        "uniform_negative_ce": {
            "label": "Uniform + LogLikeLihood",
            "color": "tab:orange",
        },
    }

    for item in grouped_items:
        title = (
            f"{item['model']} | eps={item['eps']} | "
            f"{item['explain_method']} | {item['algorithm']}"
        )

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
        for combo_name, config in style.items():
            asr_curve = item[combo_name]["asr_curve"]
            sal_curve = item[combo_name]["saliency_curve"]

            axes[0].plot(
                range(1, len(asr_curve) + 1),
                asr_curve,
                label=config["label"],
                color=config["color"],
                linewidth=2.0,
            )
            axes[1].plot(
                range(1, len(sal_curve) + 1),
                sal_curve,
                label=config["label"],
                color=config["color"],
                linewidth=2.0,
            )

        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("ASR")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8)

        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Mean saliency loss")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8)

        fig.suptitle(title)
        fig.tight_layout()

        file_name = (
            f"{item['model']}__eps-{item['eps']}"
            f"__exp-{item['explain_method']}__algo-{item['algorithm']}.png"
        )
        fig.savefig(plot_dir / file_name, dpi=170)
        plt.close(fig)


def _plot_overall_grouped_fourway(overall: Dict[str, object], output_dir: Path) -> None:
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        return

    plot_dir = output_dir / "plots" / "compare_grouped"
    plot_dir.mkdir(parents=True, exist_ok=True)

    style = {
        "saliency_guided_negative_ce": {
            "label": "Saliency Guided + LogLikeLihood",
            "color": "tab:red",
        },
        "uniform_margin": {
            "label": "Uniform + Margin",
            "color": "tab:blue",
        },
        "saliency_guided_margin": {
            "label": "Saliency Guided + Margin",
            "color": "tab:green",
        },
        "uniform_negative_ce": {
            "label": "Uniform + LogLikeLihood",
            "color": "tab:orange",
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    for combo_name, config in style.items():
        asr_curve = overall[combo_name]["asr_curve"]
        sal_curve = overall[combo_name]["saliency_curve"]

        axes[0].plot(
            range(1, len(asr_curve) + 1),
            asr_curve,
            label=config["label"],
            color=config["color"],
            linewidth=2.2,
        )
        axes[1].plot(
            range(1, len(sal_curve) + 1),
            sal_curve,
            label=config["label"],
            color=config["color"],
            linewidth=2.2,
        )

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("ASR")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Mean saliency loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle(f"compare_grouped overall average (groups={overall['num_groups']})")
    fig.tight_layout()
    fig.savefig(plot_dir / "overall__compare_grouped_fourway.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    root_dir = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    if args.all_runs_json:
        all_runs_path = Path(args.all_runs_json)
        if not all_runs_path.exists():
            raise FileNotFoundError(f"all_runs JSON not found: {all_runs_path}")
        all_runs = _load_runs_from_json(all_runs_path)
    else:
        all_runs = _load_all_runs(
            root_dir,
            target_algorithm=target_algo,
            target_explain_method=target_explain_method,
            target_w_m=target_w_m,
            target_w_s=target_w_s,
            target_pairs=target_pairs,
        )

    if not all_runs:
        raise ValueError(f"No valid run found under: {root_dir}")

    filtered = all_runs
    if target_algo is not None:
        filtered = [r for r in filtered if r.algorithm == target_algo]

    if target_explain_method is not None:
        filtered = [r for r in filtered if r.explain_method == target_explain_method]

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
    overall_compare_loss = _build_overall_compare_loss(loss_pairs)
    overall_compare_init = _build_overall_compare_init(init_pairs)
    grouped_fourway = _build_grouped_fourway_curves(filtered)
    overall_grouped_fourway = _build_overall_grouped_fourway(grouped_fourway)
    latex_rows = _build_latex_rows(filtered)

    _save_json(output_dir / "all_runs.json", [r.__dict__ | {"run_dir": str(r.run_dir)} for r in filtered])
    _save_json(output_dir / "compare_loss_curves.json", loss_pairs)
    _save_json(output_dir / "compare_init_curves.json", init_pairs)
    _save_json(output_dir / "compare_loss_curves_overall.json", overall_compare_loss)
    _save_json(output_dir / "compare_init_curves_overall.json", overall_compare_init)
    _save_json(output_dir / "compare_grouped_fourway_curves.json", grouped_fourway)
    _save_json(output_dir / "compare_grouped_fourway_overall.json", overall_grouped_fourway)
    _save_json(output_dir / "latex_rows.json", latex_rows)

    if args.make_plots:
        print(
            "[INFO] --make-plots is ignored in statis_results.py. "
            "Use evaluate_script/plot_from_all_runs.py with --make-plots instead."
        )

    print(f"Loaded runs: {len(all_runs)}")
    print(f"Runs after filter: {len(filtered)}")
    print(f"Loss pairs: {len(loss_pairs)}")
    print(f"Init pairs: {len(init_pairs)}")
    print(f"Overall compare_loss pairs: {overall_compare_loss['num_pairs']}")
    print(f"Overall compare_init pairs: {overall_compare_init['num_pairs']}")
    print(f"Grouped fourway charts: {len(grouped_fourway)}")
    print(f"Overall grouped fourway charts: {overall_grouped_fourway['num_groups']}")
    total_spearman_failed = sum(r.spearman_failed_samples for r in filtered)
    print(f"Spearman failed samples: {total_spearman_failed}")
    print("Samples per folder (ok/total):")
    for run in sorted(filtered, key=lambda r: str(r.run_dir)):
        print(f"  - {run.run_dir}: {run.num_samples_ok}/{run.num_samples_total}")
    print(f"Output dir: {output_dir}")

    if args.print_latex:
        print("\n=== LaTeX Rows ===")
        for row in latex_rows:
            print(row)


if __name__ == "__main__":
    main()




