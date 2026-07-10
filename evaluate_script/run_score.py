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


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_mean(values: List[float]) -> float:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


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
        raise ValueError("Spearman inputs must have same number of elements")
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


def _read_history_from_text(history_path: Path) -> Tuple[List[float], List[float]]:
    margin: List[float] = []
    saliency: List[float] = []

    if not history_path.exists():
        return margin, saliency

    with open(history_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = re.split(r"[\s,]+", line)
            if len(parts) >= 2:
                margin.append(_safe_float(parts[0]))
                saliency.append(_safe_float(parts[1]))
            elif len(parts) == 1:
                saliency.append(_safe_float(parts[0]))

    return margin, saliency


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


def _compute_spearman_for_sample(
    result: Dict[str, object],
    report_path: Path,
) -> Tuple[Optional[float], Optional[Dict[str, object]]]:
    candidates = _resolve_output_dir_candidates(result=result, report_path=report_path)

    clean_map_path = None
    adv_map_path = None

    for output_dir in candidates:
        clean_candidate = output_dir / "clean_map.png"
        adv_candidate = output_dir / "adv_map.png"
        if clean_candidate.exists() and adv_candidate.exists():
            clean_map_path = clean_candidate
            adv_map_path = adv_candidate
            break

    if clean_map_path is None or adv_map_path is None:
        return None, {"reason": "missing_clean_or_adv_map"}

    clean_map = _load_gray_image_array(clean_map_path)
    adv_map = _load_gray_image_array(adv_map_path)

    corr = _spearman_rank_corr(clean_map, adv_map)
    if corr is None or math.isnan(corr):
        return corr, {"reason": "nan_spearman"}

    return corr, None


def _curve_mean_with_last_padding(histories: List[List[float]]) -> List[float]:
    valid = [h for h in histories if h]
    if not valid:
        return []

    max_len = max(len(h) for h in valid)
    means: List[float] = []
    for i in range(max_len):
        values = []
        for history in valid:
            values.append(history[i] if i < len(history) else history[-1])
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


def _load_results_from_run_folder(run_dir: Path) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    summary_paths = list(run_dir.glob("*/*/summary.json")) + list(run_dir.glob("*/*/summarize.json"))
    seen_paths = set()

    for summary_path in sorted(summary_paths):
        key = str(summary_path)
        if key in seen_paths:
            continue
        seen_paths.add(key)

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        if "output_dir" not in payload:
            payload["output_dir"] = str(summary_path.parent)

        if not isinstance(payload.get("history_saliency"), list) or len(payload.get("history_saliency", [])) == 0:
            _, saliency_hist = _read_history_from_text(summary_path.parent / "history_scores.txt")
            if not saliency_hist:
                _, saliency_hist = _read_history_from_text(summary_path.parent / "history.txt")
            if saliency_hist:
                payload["history_saliency"] = saliency_hist

        results.append(payload)

    return results


def _extract_run_stats(
    run_dir: Path,
    model_name: str,
    target_algorithm: Optional[str] = None,
    target_explain_method: Optional[str] = None,
    target_w_m: Optional[float] = None,
    target_w_s: Optional[float] = None,
    target_pairs: Optional[List[Tuple[float, float]]] = None,
) -> Optional[RunStats]:
    report: Dict[str, object] = {
        "model": model_name,
        "approach": run_dir.name,
        "results": _load_results_from_run_folder(run_dir),
    }

    approach = str(report.get("approach", run_dir.name))
    meta = _parse_approach(approach)
    strategy = str(meta["strategy"])
    eps = meta["eps"]
    explain_method = str(meta["explain_method"])
    algorithm = str(meta["algorithm"])
    loss_type = str(meta["loss_type"])
    w_m = meta["w_m"]
    w_s = meta["w_s"]
    if eps is None:
        eps = 0

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
    if not isinstance(all_results, list):
        all_results = []
    results = [r for r in all_results if isinstance(r, dict) and r.get("status") == "ok"]

    if model_name in {"unknown", ""}:
        for item in all_results:
            if isinstance(item, dict) and item.get("model"):
                model_name = str(item.get("model"))
                break

    if not results:
        return RunStats(
            run_dir=run_dir,
            model=model_name,
            strategy=strategy,
            eps=int(float(eps)),
            explain_method=explain_method,
            algorithm=algorithm,
            loss_type=loss_type,
            w_m=w_m,
            w_s=w_s,
            asr=0.0,
            spearman=float("nan"),
            spearman_failed_samples=0,
            num_samples_total=len(all_results),
            num_samples_ok=0,
            asr_curve=[],
            saliency_curve=[],
        )

    first_success_iters: List[Optional[int]] = []
    saliency_histories: List[List[float]] = []
    spearman_scores: List[float] = []
    spearman_failed_samples = 0
    max_iter = 0
    final_success_count = 0

    for result in tqdm(results, desc=f"samples {model_name}/{run_dir.name}", unit="img", leave=False):
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
        sal_curve = [_safe_float(v) for v in history_sal] if isinstance(history_sal, list) else []
        saliency_histories.append(sal_curve)
        if len(sal_curve) > max_iter:
            max_iter = len(sal_curve)

        if first_iter is None and final_success:
            first_iter = len(sal_curve) if sal_curve else 1
        first_success_iters.append(first_iter)

        corr, err = _compute_spearman_for_sample(result=result, report_path=run_dir / "batch_report.json")
        if err is not None:
            spearman_failed_samples += 1
        if corr is not None:
            spearman_scores.append(corr)

    asr = final_success_count / len(results)
    spearman = _safe_mean(spearman_scores)
    asr_curve = _build_asr_curve(total_samples=len(results), first_success_iters=first_success_iters, max_iter=max_iter)
    saliency_curve = _curve_mean_with_last_padding(saliency_histories)

    return RunStats(
        run_dir=run_dir,
        model=model_name,
        strategy=strategy,
        eps=int(float(eps)),
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

    if _has_summary_files(root_dir):
        stats = _extract_run_stats(
            run_dir=root_dir,
            model_name="unknown",
            target_algorithm=target_algorithm,
            target_explain_method=target_explain_method,
            target_w_m=target_w_m,
            target_w_s=target_w_s,
            target_pairs=target_pairs,
        )
        if stats is not None:
            all_runs.append(stats)
        return all_runs

    direct_children = [d for d in sorted(root_dir.iterdir()) if d.is_dir()]
    direct_run_dirs = [d for d in direct_children if _has_summary_files(d)]
    if direct_run_dirs:
        for run_dir in tqdm(direct_run_dirs, desc="runs", unit="run"):
            stats = _extract_run_stats(
                run_dir=run_dir,
                model_name="unknown",
                target_algorithm=target_algorithm,
                target_explain_method=target_explain_method,
                target_w_m=target_w_m,
                target_w_s=target_w_s,
                target_pairs=target_pairs,
            )
            if stats is not None:
                all_runs.append(stats)
        return all_runs

    model_dirs = [d for d in sorted(root_dir.iterdir()) if d.is_dir()]
    for model_dir in tqdm(model_dirs, desc="models", unit="model"):
        model_name = model_dir.name
        run_dirs = [d for d in sorted(model_dir.iterdir()) if d.is_dir()]
        for run_dir in tqdm(run_dirs, desc=f"runs {model_name}", unit="run", leave=False):
            stats = _extract_run_stats(
                run_dir=run_dir,
                model_name=model_name,
                target_algorithm=target_algorithm,
                target_explain_method=target_explain_method,
                target_w_m=target_w_m,
                target_w_s=target_w_s,
                target_pairs=target_pairs,
            )
            if stats is not None:
                all_runs.append(stats)

    return all_runs


def _curve_mean(curves: List[List[float]]) -> List[float]:
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
        key=lambda r: (-_safe_float(r.asr, -1.0), _safe_float(r.spearman, 999.0), str(r.run_dir)),
    )[0]


def _build_latex_rows(runs: List[RunStats]) -> List[str]:
    rows: List[str] = []
    grouped = _group_runs(runs, key_fn=lambda r: (r.model, r.eps, r.strategy, r.explain_method, r.algorithm))

    for key in sorted(grouped.keys(), key=lambda x: (str(x[0]), int(x[1]), str(x[2]), str(x[3]), str(x[4]))):
        model, eps, strategy, _, _ = key
        group = grouped[key]

        margin = _choose_best_by_asr_then_spearman([g for g in group if g.loss_type == "margin_loss"])
        ce = _choose_best_by_asr_then_spearman([g for g in group if g.loss_type == "negative_cross_entropy_saliency"])
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
    grouped = _group_runs(runs, key_fn=lambda r: (r.model, r.eps, r.strategy, r.explain_method, r.algorithm))
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
                    "asr_curve": _curve_mean([r.asr_curve for r in margin_runs]),
                    "saliency_curve": _curve_mean([r.saliency_curve for r in margin_runs]),
                    "final_asr": sum(r.asr for r in margin_runs) / len(margin_runs),
                },
                "negative_cross_entropy_saliency": {
                    "asr_curve": _curve_mean([r.asr_curve for r in ce_runs]),
                    "saliency_curve": _curve_mean([r.saliency_curve for r in ce_runs]),
                    "final_asr": sum(r.asr for r in ce_runs) / len(ce_runs),
                },
            }
        )

    return output


def _build_pair_curves_init(runs: List[RunStats]) -> List[Dict[str, object]]:
    grouped = _group_runs(runs, key_fn=lambda r: (r.model, r.eps, r.loss_type, r.explain_method, r.algorithm))
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
                    "asr_curve": _curve_mean([r.asr_curve for r in saliency_runs]),
                    "saliency_curve": _curve_mean([r.saliency_curve for r in saliency_runs]),
                    "final_asr": sum(r.asr for r in saliency_runs) / len(saliency_runs),
                },
                "uniform": {
                    "asr_curve": _curve_mean([r.asr_curve for r in uniform_runs]),
                    "saliency_curve": _curve_mean([r.saliency_curve for r in uniform_runs]),
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
            "asr_curve": _curve_mean(margin_asr_curves),
            "saliency_curve": _curve_mean(margin_sal_curves),
            "final_asr": _safe_mean(margin_final_asr),
        },
        "negative_cross_entropy_saliency": {
            "asr_curve": _curve_mean(ce_asr_curves),
            "saliency_curve": _curve_mean(ce_sal_curves),
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
            "asr_curve": _curve_mean(sal_asr_curves),
            "saliency_curve": _curve_mean(sal_sal_curves),
            "final_asr": _safe_mean(sal_final_asr),
        },
        "uniform": {
            "asr_curve": _curve_mean(uni_asr_curves),
            "saliency_curve": _curve_mean(uni_sal_curves),
            "final_asr": _safe_mean(uni_final_asr),
        },
    }


def _build_grouped_fourway_curves(runs: List[RunStats]) -> List[Dict[str, object]]:
    grouped = _group_runs(runs, key_fn=lambda r: (r.model, r.eps, r.explain_method, r.algorithm))
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
                "asr_curve": _curve_mean([r.asr_curve for r in selected]),
                "saliency_curve": _curve_mean([r.saliency_curve for r in selected]),
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
            "asr_curve": _curve_mean(asr_curves),
            "saliency_curve": _curve_mean(sal_curves),
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
        if isinstance(item, dict):
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
        title = f"{item['model']} | eps={item['eps']} | {item['strategy']} | {item['explain_method']} | {item['algorithm']}"

        margin_asr = item["margin_loss"]["asr_curve"]
        ce_asr = item["negative_cross_entropy_saliency"]["asr_curve"]
        margin_sal = item["margin_loss"]["saliency_curve"]
        ce_sal = item["negative_cross_entropy_saliency"]["saliency_curve"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

        axes[0].plot(range(1, len(margin_asr) + 1), margin_asr, label="margin_loss", linewidth=2)
        axes[0].plot(range(1, len(ce_asr) + 1), ce_asr, label="LogLikeLihood", linewidth=2)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("ASR")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(range(1, len(margin_sal) + 1), margin_sal, label="margin_loss", linewidth=2)
        axes[1].plot(range(1, len(ce_sal) + 1), ce_sal, label="LogLikeLihood", linewidth=2)
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
        title = f"{item['model']} | eps={item['eps']} | {item['loss_type']} | {item['explain_method']} | {item['algorithm']}"

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
        "saliency_guided_negative_ce": {"label": "Saliency Guided + LogLikeLihood", "color": "tab:red"},
        "uniform_margin": {"label": "Uniform + Margin", "color": "tab:blue"},
        "saliency_guided_margin": {"label": "Saliency Guided + Margin", "color": "tab:green"},
        "uniform_negative_ce": {"label": "Uniform + LogLikeLihood", "color": "tab:orange"},
    }

    for item in grouped_items:
        title = f"{item['model']} | eps={item['eps']} | {item['explain_method']} | {item['algorithm']}"

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
        for combo_name, config in style.items():
            asr_curve = item[combo_name]["asr_curve"]
            sal_curve = item[combo_name]["saliency_curve"]

            axes[0].plot(range(1, len(asr_curve) + 1), asr_curve, label=config["label"], color=config["color"], linewidth=2.0)
            axes[1].plot(range(1, len(sal_curve) + 1), sal_curve, label=config["label"], color=config["color"], linewidth=2.0)

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

        file_name = f"{item['model']}__eps-{item['eps']}__exp-{item['explain_method']}__algo-{item['algorithm']}.png"
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
        "saliency_guided_negative_ce": {"label": "Saliency Guided + LogLikeLihood", "color": "tab:red"},
        "uniform_margin": {"label": "Uniform + Margin", "color": "tab:blue"},
        "saliency_guided_margin": {"label": "Saliency Guided + Margin", "color": "tab:green"},
        "uniform_negative_ce": {"label": "Uniform + LogLikeLihood", "color": "tab:orange"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    for combo_name, config in style.items():
        asr_curve = overall[combo_name]["asr_curve"]
        sal_curve = overall[combo_name]["saliency_curve"]

        axes[0].plot(range(1, len(asr_curve) + 1), asr_curve, label=config["label"], color=config["color"], linewidth=2.2)
        axes[1].plot(range(1, len(sal_curve) + 1), sal_curve, label=config["label"], color=config["color"], linewidth=2.2)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score run folders by ASR + Spearman from per-sample summaries and "
            "generate grouped comparison curves and LaTeX rows."
        )
    )
    parser.add_argument("--result-root", type=str, default="server_run/server_run", help="Root folder with model/run directories")
    parser.add_argument("--all-runs-json", type=str, default=None, help="Optional precomputed all_runs.json path")
    parser.add_argument("--output-dir", type=str, default="evaluate_script/stats_outputs", help="Directory to save output JSON/plots")
    parser.add_argument("--algorithm", type=str, default="all", help="Filter algorithm (weighted_sum_ga or nsgaii). Use all to keep all")
    parser.add_argument("--explain-method", type=str, default="all", help="Filter explain method. Use all to keep all")
    parser.add_argument("--w-m", type=str, default="all", help="Filter by w_m. Use numeric value or all")
    parser.add_argument("--w-c", type=str, default="all", help="Backward-compatible alias of --w-s")
    parser.add_argument("--w-s", type=str, default="all", help="Filter by w_s. Use numeric value or all")
    parser.add_argument("--weight-pairs", type=str, default=None, help="Comma-separated wm:ws pairs, e.g. 1:0,0.5:0.5")
    parser.add_argument(
        "--compare-uniform-loss-only",
        action="store_true",
        help=(
            "Keep only uniform strategy and only export/plot loss comparison "
            "(margin_loss vs LogLikeLihood)"
        ),
    )
    parser.add_argument("--print-latex", action="store_true", help="Print LaTeX rows to stdout")
    parser.add_argument("--make-plots", action="store_true", help="Generate PNG plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result_root = Path(args.result_root)
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
            result_root,
            target_algorithm=target_algo,
            target_explain_method=target_explain_method,
            target_w_m=target_w_m,
            target_w_s=target_w_s,
            target_pairs=target_pairs,
        )

    if not all_runs:
        raise ValueError(f"No valid run found under: {result_root}")

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

    if args.compare_uniform_loss_only:
        filtered = [r for r in filtered if r.strategy == "uniform"]

    if not filtered:
        raise ValueError("No run left after filtering")

    loss_pairs = _build_pair_curves_loss(filtered)
    overall_compare_loss = _build_overall_compare_loss(loss_pairs)
    latex_rows = _build_latex_rows(filtered)

    init_pairs: List[Dict[str, object]] = []
    grouped_fourway: List[Dict[str, object]] = []
    overall_compare_init: Dict[str, object] = {
        "num_pairs": 0,
        "averaged_over": "disabled_by_compare_uniform_loss_only",
    }
    overall_grouped_fourway: Dict[str, object] = {
        "num_groups": 0,
        "averaged_over": "disabled_by_compare_uniform_loss_only",
    }

    if not args.compare_uniform_loss_only:
        init_pairs = _build_pair_curves_init(filtered)
        grouped_fourway = _build_grouped_fourway_curves(filtered)
        overall_compare_init = _build_overall_compare_init(init_pairs)
        overall_grouped_fourway = _build_overall_grouped_fourway(grouped_fourway)

    _save_json(output_dir / "all_runs.json", [r.__dict__ | {"run_dir": str(r.run_dir)} for r in filtered])
    _save_json(output_dir / "compare_loss_curves.json", loss_pairs)
    _save_json(output_dir / "compare_loss_curves_overall.json", overall_compare_loss)
    if not args.compare_uniform_loss_only:
        _save_json(output_dir / "compare_init_curves.json", init_pairs)
        _save_json(output_dir / "compare_init_curves_overall.json", overall_compare_init)
        _save_json(output_dir / "compare_grouped_fourway_curves.json", grouped_fourway)
        _save_json(output_dir / "compare_grouped_fourway_overall.json", overall_grouped_fourway)
    _save_json(output_dir / "latex_rows.json", latex_rows)

    if args.make_plots:
        _plot_pair_curves_loss(loss_pairs, output_dir)
        _plot_overall_compare_loss(overall_compare_loss, output_dir)
        if not args.compare_uniform_loss_only:
            _plot_pair_curves_init(init_pairs, output_dir)
            _plot_overall_compare_init(overall_compare_init, output_dir)
            _plot_grouped_fourway(grouped_fourway, output_dir)
            _plot_overall_grouped_fourway(overall_grouped_fourway, output_dir)

    print(f"Loaded runs: {len(all_runs)}")
    print(f"Runs after filter: {len(filtered)}")
    print(f"Loss pairs: {len(loss_pairs)}")
    if args.compare_uniform_loss_only:
        print("Mode: compare_uniform_loss_only")
    else:
        print(f"Init pairs: {len(init_pairs)}")
        print(f"Grouped fourway charts: {len(grouped_fourway)}")
    print(f"Output dir: {output_dir}")

    if args.print_latex:
        print("\n=== LaTeX Rows ===")
        for row in latex_rows:
            print(row)


if __name__ == "__main__":
    main()
