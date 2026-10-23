import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

try:
    from scipy.stats import ttest_ind_from_stats
except ImportError:
    ttest_ind_from_stats = None

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
    objective_curve: List[float]
    seed: str = "default"
    h_score: float = float("nan")


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
    if lower in {"", "default", "weighted_sum", "weighted_sum_ga", "wga", "ga"}:
        return "weighted_sum_ga"
    if lower in {"nsga", "nsgaii", "nsga-ii", "nsga2", "nsga_2"}:
        return "nsgaii"
    return lower


def _normalize_model_name(name: str) -> str:
    return str(name or "").strip().lower()


def _parse_approach(approach: str) -> Dict[str, object]:
    fields: Dict[str, object] = {
        "strategy": "unknown",
        "eps": None,
        "explain_method": "unknown",
        "algorithm": "weighted_sum_ga",
        "loss_type": "margin_loss",
        "w_m": None,
        "w_s": None,
        "seed": None,
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
        elif token.startswith("seed-"):
            fields["seed"] = token[len("seed-") :]

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


def _parse_optional_int_arg(raw: str, arg_name: str) -> Optional[int]:
    text = str(raw or "").strip().lower()
    if text in {"", "all"}:
        return None
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"Invalid value for {arg_name}: {raw}. Use integer or 'all'.") from exc


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


def _is_wm0_ws1(w_m: Optional[float], w_s: Optional[float]) -> bool:
    return _is_close(w_m, 0.0) and _is_close(w_s, 1.0)


def _expand_equivalent_losses_for_wm0_ws1(runs: List[RunStats]) -> List[RunStats]:
    grouped = _group_runs(
        runs,
        key_fn=lambda r: (r.model, r.seed, r.eps, r.strategy, r.explain_method, r.algorithm, r.w_m, r.w_s),
    )

    expanded: List[RunStats] = list(runs)
    for key, group in grouped.items():
        _, _, _, _, _, _, w_m, w_s = key
        if not _is_wm0_ws1(w_m, w_s):
            continue

        has_margin = any(r.loss_type == "margin_loss" for r in group)
        has_ce = any(r.loss_type == "negative_cross_entropy_saliency" for r in group)
        if has_margin and has_ce:
            continue

        source = group[0]
        mirrored_loss = "negative_cross_entropy_saliency" if has_margin else "margin_loss"
        expanded.append(
            RunStats(
                run_dir=source.run_dir,
                model=source.model,
                strategy=source.strategy,
                eps=source.eps,
                explain_method=source.explain_method,
                algorithm=source.algorithm,
                loss_type=mirrored_loss,
                w_m=source.w_m,
                w_s=source.w_s,
                asr=source.asr,
                spearman=source.spearman,
                spearman_failed_samples=source.spearman_failed_samples,
                num_samples_total=source.num_samples_total,
                num_samples_ok=source.num_samples_ok,
                asr_curve=source.asr_curve,
                saliency_curve=source.saliency_curve,
                objective_curve=source.objective_curve,
                seed=source.seed,
                h_score=source.h_score,
            )
        )

    return expanded


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

        margin_missing = not isinstance(payload.get("history_margin"), list) or len(payload.get("history_margin", [])) == 0
        saliency_missing = not isinstance(payload.get("history_saliency"), list) or len(payload.get("history_saliency", [])) == 0
        if margin_missing or saliency_missing:
            margin_hist, saliency_hist = _read_history_from_text(summary_path.parent / "history_scores.txt")
            if not margin_hist and not saliency_hist:
                margin_hist, saliency_hist = _read_history_from_text(summary_path.parent / "history.txt")
            if margin_missing and margin_hist:
                payload["history_margin"] = margin_hist
            if saliency_missing and saliency_hist:
                payload["history_saliency"] = saliency_hist

        results.append(payload)

    return results


def _resolve_run_context(run_dir: Path, root_dir: Path) -> Tuple[str, str, str]:
    approach_name = run_dir.name
    meta = _parse_approach(approach_name)

    algo_from_meta = meta.get("algorithm")
    seed_from_meta = meta.get("seed")

    ancestors: List[str] = []
    curr = run_dir.parent
    try:
        resolved_root = root_dir.resolve()
        while True:
            ancestors.append(curr.name)
            if curr.resolve() == resolved_root or curr == curr.parent:
                break
            curr = curr.parent
    except Exception:
        curr = run_dir.parent
        for _ in range(5):
            ancestors.append(curr.name)
            if curr == root_dir or curr == curr.parent:
                break
            curr = curr.parent

    # ancestors[0] is the immediate parent: model folder
    inferred_model = ancestors[0] if ancestors else "unknown"

    inferred_seed = None
    inferred_algo = None

    for anc in ancestors[1:]:
        norm_algo = _normalize_algorithm(anc)
        if norm_algo in {"weighted_sum_ga", "nsgaii"}:
            if inferred_algo is None:
                inferred_algo = norm_algo
        else:
            if inferred_seed is None:
                inferred_seed = anc

    if algo_from_meta and algo_from_meta != "weighted_sum_ga":
        final_algo = algo_from_meta
    elif inferred_algo:
        final_algo = inferred_algo
    else:
        final_algo = algo_from_meta or "weighted_sum_ga"

    final_seed = seed_from_meta or inferred_seed or "default"
    final_model = inferred_model

    return final_model, final_algo, str(final_seed)


def _discover_all_run_dirs(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        return []

    if _has_summary_files(root_dir):
        return [root_dir]

    run_dirs: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        p = Path(dirpath)
        if p.name.startswith("strategy-") and _has_summary_files(p):
            run_dirs.append(p)
            dirnames.clear()
        elif _has_summary_files(p):
            has_child_run = any(
                (p / child).is_dir() and _has_summary_files(p / child)
                for child in dirnames
            )
            if not has_child_run:
                run_dirs.append(p)
                dirnames.clear()

    return sorted(run_dirs)


def _extract_run_stats(
    run_dir: Path,
    model_name: str,
    target_model: Optional[str] = None,
    inferred_algorithm: Optional[str] = None,
    inferred_seed: Optional[str] = None,
    target_eps: Optional[int] = None,
    target_algorithm: Optional[str] = None,
    target_explain_method: Optional[str] = None,
    target_w_m: Optional[float] = None,
    target_w_s: Optional[float] = None,
    target_pairs: Optional[List[Tuple[float, float]]] = None,
    target_seed: Optional[str] = None,
    min_sample_ratio: float = 0.5,
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
    seed = str(meta.get("seed") or inferred_seed or "default")

    if inferred_algorithm and (algorithm == "weighted_sum_ga" or not algorithm):
        algorithm = inferred_algorithm

    if eps is None:
        eps = 0
    eps_int = int(float(eps))

    if target_eps is not None and eps_int != target_eps:
        return None

    if target_algorithm is not None and algorithm != target_algorithm:
        return None
    if target_seed is not None and target_seed.lower() != "all" and str(seed) != str(target_seed):
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

    if target_model is not None and _normalize_model_name(model_name) != _normalize_model_name(target_model):
        return None

    # If no successful samples, or missing more than (1 - min_sample_ratio) of sample files, skip run
    if not results:
        return None

    if min_sample_ratio > 0.0 and len(all_results) > 0 and (len(results) / len(all_results)) < min_sample_ratio:
        return None

    first_success_iters: List[Optional[int]] = []
    saliency_histories: List[List[float]] = []
    objective_histories: List[List[float]] = []
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

        if loss_type == "margin_loss":
            history_obj = result.get("history_margin", [])
        else:
            history_obj = (
                result.get("history_attack_objective")
                or result.get("history_objective")
                or result.get("history_ce")
                or result.get("history_cross_entropy")
                or result.get("history_margin")
                or []
            )
        obj_curve = [_safe_float(v) for v in history_obj] if isinstance(history_obj, list) else []
        objective_histories.append(obj_curve)

        if len(sal_curve) > max_iter:
            max_iter = len(sal_curve)
        if len(obj_curve) > max_iter:
            max_iter = len(obj_curve)

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
    h_score = _h_score(asr, spearman)
    asr_curve = _build_asr_curve(total_samples=len(results), first_success_iters=first_success_iters, max_iter=max_iter)
    saliency_curve = _curve_mean_with_last_padding(saliency_histories)
    objective_curve = _curve_mean_with_last_padding(objective_histories)

    return RunStats(
        run_dir=run_dir,
        model=model_name,
        strategy=strategy,
        eps=eps_int,
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
        objective_curve=objective_curve,
        seed=seed,
        h_score=h_score,
    )


def _load_all_runs(
    root_dir: Path,
    target_model: Optional[str] = None,
    target_eps: Optional[int] = None,
    target_algorithm: Optional[str] = None,
    target_explain_method: Optional[str] = None,
    target_w_m: Optional[float] = None,
    target_w_s: Optional[float] = None,
    target_pairs: Optional[List[Tuple[float, float]]] = None,
    target_seed: Optional[str] = None,
    min_sample_ratio: float = 0.5,
    expected_samples: Optional[int] = None,
) -> List[RunStats]:
    all_runs: List[RunStats] = []
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"root dir not found: {root_dir}")

    run_dirs = _discover_all_run_dirs(root_dir)
    print(f"Discovered {len(run_dirs)} candidate run directories under: {root_dir}")

    raw_runs: List[RunStats] = []
    for run_dir in tqdm(run_dirs, desc="extracting runs", unit="run"):
        inferred_model, inferred_algo, inferred_seed = _resolve_run_context(run_dir, root_dir)

        if target_model is not None and inferred_model != "unknown" and _normalize_model_name(inferred_model) != _normalize_model_name(target_model):
            continue

        if target_algorithm is not None and inferred_algo != target_algorithm:
            continue

        if target_seed is not None and target_seed.lower() != "all" and str(inferred_seed) != str(target_seed):
            continue

        stats = _extract_run_stats(
            run_dir=run_dir,
            model_name=inferred_model,
            target_model=target_model,
            inferred_algorithm=inferred_algo,
            inferred_seed=inferred_seed,
            target_eps=target_eps,
            target_algorithm=target_algorithm,
            target_explain_method=target_explain_method,
            target_w_m=target_w_m,
            target_w_s=target_w_s,
            target_pairs=target_pairs,
            target_seed=target_seed,
            min_sample_ratio=min_sample_ratio,
        )
        if stats is not None:
            raw_runs.append(stats)

    # Determine expected samples per model
    expected_per_model: Dict[str, int] = {}
    for r in raw_runs:
        m_norm = _normalize_model_name(r.model)
        expected_per_model[m_norm] = max(expected_per_model.get(m_norm, 0), r.num_samples_ok)

    skipped_incomplete = 0
    for r in raw_runs:
        m_norm = _normalize_model_name(r.model)
        exp = expected_samples if expected_samples is not None else expected_per_model.get(m_norm, 0)
        if min_sample_ratio > 0.0 and exp > 0 and (r.num_samples_ok / exp) < min_sample_ratio:
            skipped_incomplete += 1
            continue
        all_runs.append(r)

    if skipped_incomplete > 0:
        print(f"Filtered out {skipped_incomplete} incomplete runs with < {min_sample_ratio*100:.0f}% of expected samples.")

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


def _h_score(asr: float, sro: float) -> float:
    inv_sro = 1.0 - float(sro)
    denom = float(asr) + inv_sro
    if abs(denom) < 1e-12:
        return float("nan")
    return (2.0 * float(asr) * inv_sro) / denom


@dataclass
class MetricStat:
    mean: float
    std: float
    count: int
    values: List[float]


def _calc_stat(values: List[float]) -> MetricStat:
    valid = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not valid:
        return MetricStat(mean=float("nan"), std=float("nan"), count=0, values=[])
    if len(valid) == 1:
        return MetricStat(mean=valid[0], std=0.0, count=1, values=valid)
    mean_val = sum(valid) / len(valid)
    var = sum((x - mean_val) ** 2 for x in valid) / (len(valid) - 1)
    std_val = math.sqrt(max(0.0, var))
    return MetricStat(mean=mean_val, std=std_val, count=len(valid), values=valid)


def _match_method(run: RunStats) -> Optional[Tuple[str, str]]:
    """
    Returns (method_id, latex_label) for a RunStats instance, or None if unrecognized.
    Standard methods:
      - 'ga_0.0' : GA (\\lambda=0.0)
      - 'ga_0.5' : GA (\\lambda=0.5)
      - 'ga_1.0' : GA (\\lambda=1.0)
      - 'nsgaii' : \\textbf{NSGA-II}
    """
    algo = run.algorithm
    if algo == "nsgaii":
        return ("nsgaii", r"\textbf{NSGA-II}")
    if algo in ("weighted_sum_ga", "ga"):
        if run.w_m is not None:
            try:
                wm = float(run.w_m)
            except (ValueError, TypeError):
                return ("ga_unknown", "GA")
            if abs(wm - 0.0) < 1e-4:
                return ("ga_0.0", r"GA ($\lambda=0.0$)")
            elif abs(wm - 0.5) < 1e-4:
                return ("ga_0.5", r"GA ($\lambda=0.5$)")
            elif abs(wm - 1.0) < 1e-4:
                return ("ga_1.0", r"GA ($\lambda=1.0$)")
            else:
                return (f"ga_{wm:.2f}", f"GA ($\\lambda={wm:.2f}$)")
        return ("ga_unknown", "GA")
    return (algo, algo)


def _format_model_latex_name(model_name: str) -> str:
    norm = _normalize_model_name(model_name)
    mapping = {
        "resnet18": "ResNet18",
        "resnet50": "ResNet50",
        "vgg16": "VGG16",
        "vgg19": "VGG19",
        "vit_b_32": "ViT-B-32",
        "vit-b-32": "ViT-B-32",
        "densenet121": "DenseNet121",
        "inception_v3": "InceptionV3",
    }
    return mapping.get(norm, model_name)


def _sort_model_keys(models: List[str]) -> List[str]:
    priority = {"resnet18": 0, "vgg16": 1, "vit_b_32": 2, "vit-b-32": 2}
    return sorted(models, key=lambda m: (priority.get(_normalize_model_name(m), 99), m))


def _format_explain_latex_name(explain_method: Optional[str]) -> str:
    if not explain_method:
        return "Simple Gradient"
    norm = explain_method.lower().strip()
    mapping = {
        "simple_gradient": "Simple Gradient",
        "input_gradient": r"Input $\times$ Gradient",
        "grad_cam": "Grad-CAM",
        "raw_attention": "Raw Attention",
        "integrated_gradients": "Integrated Gradients",
        "attention_grad": "Attention Grad",
    }
    return mapping.get(norm, explain_method.replace("_", " ").title())


def _clean_explain_method_tag(explain_method: Optional[str]) -> str:
    if not explain_method:
        return "default"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", explain_method.lower().strip())


def _run_ttest(
    mean1: float,
    std1: float,
    n1: int = 5,
    mean2: float = 0.0,
    std2: float = 0.0,
    n2: int = 5,
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> Tuple[Optional[float], Optional[float]]:
    """
    Performs two-sample independent Welch's t-test between candidate 1 and candidate 2
    using scipy.stats.ttest_ind_from_stats.
    Returns (t_stat, p_value). If the test cannot be calculated, returns (None, None).
    """
    if ttest_ind_from_stats is None:
        return None, None

    if (
        math.isnan(mean1) or math.isnan(mean2)
        or math.isnan(std1) or math.isnan(std2)
        or n1 is None or n1 < 2
        or n2 is None or n2 < 2
    ):
        return None, None

    # Both std are virtually zero
    if abs(std1) < 1e-12 and abs(std2) < 1e-12:
        if abs(mean1 - mean2) < 1e-6:
            return 0.0, 1.0  # Identical, p=1.0 (not significant)
        else:
            return float("inf"), 0.0  # Strictly different with zero variance, p=0.0 (significant)

    try:
        try:
            res = ttest_ind_from_stats(
                mean1=mean1,
                std1=std1,
                nobs1=n1,
                mean2=mean2,
                std2=std2,
                nobs2=n2,
                equal_var=equal_var,
                alternative=alternative,
            )
        except TypeError:
            res = ttest_ind_from_stats(
                mean1=mean1,
                std1=std1,
                nobs1=n1,
                mean2=mean2,
                std2=std2,
                nobs2=n2,
                equal_var=equal_var,
            )
        stat_val = float(res.statistic)
        pval_val = float(res.pvalue)
        if not math.isnan(stat_val) and not math.isnan(pval_val):
            return stat_val, pval_val
    except Exception:
        pass

    return None, None


def _format_cell(
    stat: MetricStat,
    cell_format: str = "subscript",
    precision: int = 4,
    empty_cell: str = "-",
    is_asr: bool = False,
    asr_percentage: bool = False,
    is_best: bool = False,
    highlight_best: bool = True,
    best_color: str = "blue",
    best_format: str = "textbf_textcolor",
) -> str:
    if stat.count == 0 or math.isnan(stat.mean):
        return empty_cell

    mean = stat.mean * 100.0 if (is_asr and asr_percentage) else stat.mean
    std = stat.std * 100.0 if (is_asr and asr_percentage) else stat.std

    if stat.count == 1 or abs(std) < 1e-12:
        val_str = f"${mean:.{precision}f}$"
    elif cell_format == "subscript":
        val_str = f"${mean:.{precision}f}_{{\\pm {std:.{precision}f}}}$"
    elif cell_format == "subscript_raw":
        val_str = f"${mean:.{precision}f}_{{{std:.{precision}f}}}$"
    elif cell_format == "mean_only":
        val_str = f"${mean:.{precision}f}$"
    else:  # "pm" default
        val_str = f"${mean:.{precision}f} \\pm {std:.{precision}f}$"

    if is_best and highlight_best:
        if best_format == "textbf_textcolor":
            return f"\\textbf{{\\textcolor{{{best_color}}}{{{val_str}}}}}"
        elif best_format == "textcolor":
            return f"\\textcolor{{{best_color}}}{{{val_str}}}"
        elif best_format == "textbf":
            return f"\\textbf{{{val_str}}}"

    return val_str


def _aggregate_seed_stats(
    runs: List[RunStats],
    min_seed_ratio: float = 0.5,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, object]]]]]:
    """
    Groups runs across seeds and calculates mean/std for each (model, method, eps, loss_type).
    Skips reporting any cell that is missing more than (1 - min_seed_ratio) of seeds.
    Returns nested dict:
      model -> method_id -> eps (str) -> loss_type -> {
          "asr": MetricStat,
          "sro": MetricStat,
          "h_score": MetricStat,
          "seeds": List[str],
          "num_runs": int,
          "expected_seeds": int,
          "seed_ratio": float,
      }
    """
    grouped: Dict[Tuple[str, str, int, str], Dict[str, List[RunStats]]] = {}
    total_seeds_by_method: Dict[Tuple[str, str], Set[str]] = {}

    for r in runs:
        m_info = _match_method(r)
        if m_info is None:
            continue
        method_id, _ = m_info
        model_norm = _normalize_model_name(r.model)
        key = (model_norm, method_id, int(r.eps), r.loss_type)
        grouped.setdefault(key, {}).setdefault(r.seed, []).append(r)
        total_seeds_by_method.setdefault((model_norm, method_id), set()).add(r.seed)

    out: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, object]]]]] = {}

    for (model_norm, method_id, eps, loss_type), seed_map in grouped.items():
        seed_runs: List[RunStats] = []
        seeds_list: List[str] = []
        for s in sorted(seed_map.keys()):
            best = _choose_best_by_asr_then_spearman(seed_map[s])
            if best is not None:
                seed_runs.append(best)
                seeds_list.append(s)

        expected_seeds = len(total_seeds_by_method.get((model_norm, method_id), set()))
        if expected_seeds > 1 and min_seed_ratio > 0.0:
            if (len(seed_runs) / expected_seeds) < min_seed_ratio:
                # Missing > 50% seeds, pass over this cell
                continue

        asr_vals = [r.asr for r in seed_runs]
        sro_vals = [r.spearman for r in seed_runs]
        h_vals = [r.h_score for r in seed_runs]

        asr_stat = _calc_stat(asr_vals)
        sro_stat = _calc_stat(sro_vals)
        h_stat = _calc_stat(h_vals)

        m_dict = out.setdefault(model_norm, {}).setdefault(method_id, {})
        eps_dict = m_dict.setdefault(str(eps), {})
        eps_dict[loss_type] = {
            "asr": asr_stat,
            "sro": sro_stat,
            "h_score": h_stat,
            "seeds": seeds_list,
            "num_runs": len(seed_runs),
            "expected_seeds": expected_seeds,
            "seed_ratio": len(seed_runs) / max(1, expected_seeds),
        }

    return out


def _serialize_aggregated_stats(
    stats: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, object]]]]]
) -> Dict[str, object]:
    def _convert(obj):
        if isinstance(obj, MetricStat):
            return {
                "mean": None if math.isnan(obj.mean) else obj.mean,
                "std": None if math.isnan(obj.std) else obj.std,
                "count": obj.count,
                "values": [v for v in obj.values],
            }
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    return _convert(stats)


def _build_overleaf_table(
    aggregated_stats: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, object]]]]],
    models: Optional[List[str]] = None,
    explain_method: Optional[str] = None,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    cell_format: str = "subscript",
    precision: int = 4,
    empty_cell: str = "-",
    asr_percentage: bool = False,
    only_active_methods: bool = False,
    include_template_models: bool = False,
    highlight_best: bool = True,
    best_color: str = "blue",
    best_format: str = "textbf_textcolor",
    use_ttest: bool = True,
    ttest_metrics: Optional[List[str]] = None,
    ttest_alpha: float = 0.05 / 3.0,
    ttest_n: int = 5,
    ttest_alternative: str = "two-sided",
    significant_color: str = "blue",
    not_significant_color: str = "red",
    ttest_records: Optional[List[Dict[str, object]]] = None,
) -> str:
    standard_methods: List[Tuple[str, str]] = [
        ("ga_0.0", r"GA ($\lambda=0.0$)"),
        ("ga_0.5", r"GA ($\lambda=0.5$)"),
        ("ga_1.0", r"GA ($\lambda=1.0$)"),
        ("nsgaii", r"\textbf{NSGA-II}"),
    ]
    standard_models: List[str] = ["resnet18", "vgg16", "vit_b_32"]

    if ttest_metrics is None:
        ttest_metrics = ["h_score"]

    exp_latex_name = _format_explain_latex_name(explain_method)
    if caption is None:
        caption = (
            rf"Comparison between GA with different $\lambda$ values and NSGA-II ({exp_latex_name}). "
            r"$|P|=50$, $N=200$, $p_c=0.4$, $p_m=0.1$, 100 samples. "
            r"Harmonic Score$=\frac{2\,\mathrm{ASR}(1-\mathrm{SRO})}{\mathrm{ASR}+(1-\mathrm{SRO})}$."
        )
    if label is None:
        clean_tag = _clean_explain_method_tag(explain_method)
        label = f"tab:ga_vs_nsga_{clean_tag}"

    if models is not None:
        model_list = list(models)
    else:
        model_list = list(aggregated_stats.keys())

    if include_template_models:
        for sm in standard_models:
            if sm not in model_list and _normalize_model_name(sm) not in [_normalize_model_name(m) for m in model_list]:
                model_list.append(sm)

    if not model_list:
        model_list = list(standard_models)

    model_list = _sort_model_keys(model_list)

    table_header = f"""% Requires \\usepackage{{xcolor}} in your LaTeX preamble
% Column-best results are highlighted in bold:
%   For H-Score: \\textbf{{\\textcolor{{{significant_color}}}{{...}}}} indicates statistically significant advantage over runner-up (Welch's t-test, N={ttest_n}, alpha={ttest_alpha:.5f})
%                \\textbf{{\\textcolor{{{not_significant_color}}}{{...}}}} indicates advantage over runner-up is not statistically significant
%   For ASR and SRO: \\textbf{{\\textcolor{{{best_color}}}{{...}}}} indicates column-best result
\\begin{{table*}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}

\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{cc|cccccc|cccccc|cccccc}}
\\toprule

\\multirow{{3}}{{*}}{{Model}}
&
\\multirow{{3}}{{*}}{{Algorithm}}
&
\\multicolumn{{6}}{{c|}}{{{{\\epsilon=20}}}}
&
\\multicolumn{{6}}{{c|}}{{{{\\epsilon=50}}}}
&
\\multicolumn{{6}}{{c}}{{{{\\epsilon=100}}}}
\\\\

\\cmidrule(lr){{3-8}}
\\cmidrule(lr){{9-14}}
\\cmidrule(lr){{15-20}}

&
&
\\multicolumn{{3}}{{c|}}{{Margin Loss}}
&
\\multicolumn{{3}}{{c|}}{{Log-Likelihood}}
&
\\multicolumn{{3}}{{c|}}{{Margin Loss}}
&
\\multicolumn{{3}}{{c|}}{{Log-Likelihood}}
&
\\multicolumn{{3}}{{c|}}{{Margin Loss}}
&
\\multicolumn{{3}}{{c}}{{Log-Likelihood}}
\\\\

\\cmidrule(lr){{3-5}}
\\cmidrule(lr){{6-8}}
\\cmidrule(lr){{9-11}}
\\cmidrule(lr){{12-14}}
\\cmidrule(lr){{15-17}}
\\cmidrule(lr){{18-20}}

&
&
ASR$\\uparrow$
&
SRO$\\downarrow$
&
\\textbf{{H-Score}}$\\uparrow$
&
ASR$\\uparrow$
&
SRO$\\downarrow$
&
\\textbf{{H-Score}}$\\uparrow$
&
ASR$\\uparrow$
&
SRO$\\downarrow$
&
\\textbf{{H-Score}}$\\uparrow$
&
ASR$\\uparrow$
&
SRO$\\downarrow$
&
\\textbf{{H-Score}}$\\uparrow$
&
ASR$\\uparrow$
&
SRO$\\downarrow$
&
\\textbf{{H-Score}}$\\uparrow$
&
ASR$\\uparrow$
&
SRO$\\downarrow$
&
\\textbf{{H-Score}}$\\uparrow$
\\\\

\\midrule"""

    model_blocks: List[str] = []

    for model_raw in model_list:
        model_norm = _normalize_model_name(model_raw)
        model_latex = _format_model_latex_name(model_raw)

        data_for_model = aggregated_stats.get(model_norm, {})

        methods_to_use: List[Tuple[str, str]] = list(standard_methods)

        for m_id in sorted(data_for_model.keys()):
            if m_id not in [sm[0] for sm in methods_to_use]:
                methods_to_use.append((m_id, m_id))

        if only_active_methods:
            active: List[Tuple[str, str]] = []
            for m_id, m_label in methods_to_use:
                if m_id in data_for_model and any(data_for_model[m_id].values()):
                    active.append((m_id, m_label))
            if active:
                methods_to_use = active

        # Identify the best method in each column (eps, loss_type, metric)
        # and its highlight color via two-sample Welch's t-test against runner-up
        best_in_col: Dict[Tuple[int, str, str], Dict[str, str]] = {}
        if highlight_best:
            for eps in [20, 50, 100]:
                for loss_type in ["margin_loss", "negative_cross_entropy_saliency"]:
                    for metric_name in ["asr", "sro", "h_score"]:
                        candidates: List[Tuple[str, MetricStat]] = []
                        for m_id, _ in methods_to_use:
                            st = (
                                data_for_model.get(m_id, {})
                                .get(str(eps), {})
                                .get(loss_type, {})
                                .get(metric_name)
                            )
                            if (
                                st is not None
                                and isinstance(st, MetricStat)
                                and st.count > 0
                                and not math.isnan(st.mean)
                            ):
                                candidates.append((m_id, st))

                        if not candidates:
                            continue

                        # Sort: lowest for sro, highest for asr and h_score
                        if metric_name == "sro":
                            candidates.sort(key=lambda x: x[1].mean)
                        else:
                            candidates.sort(key=lambda x: -x[1].mean)

                        means = [c[1].mean for c in candidates]
                        min_v, max_v = min(means), max(means)

                        # If all methods have identical non-informative values, do not highlight
                        if len(candidates) > 1 and abs(max_v - min_v) < 1e-6:
                            continue

                        best_m_id, best_st = candidates[0]
                        should_test = use_ttest and (metric_name in ttest_metrics)

                        # If only 1 method is present in this column
                        if len(candidates) == 1:
                            best_in_col[(eps, loss_type, metric_name)] = {
                                best_m_id: significant_color if should_test else best_color
                            }
                            continue

                        runner_up_m_id, runner_up_st = candidates[1]

                        # If this metric does not undergo t-test (e.g. ASR, SRO), highlight best in best_color (blue)
                        if not should_test:
                            best_in_col[(eps, loss_type, metric_name)] = {
                                c[0]: best_color
                                for c in candidates
                                if abs(c[1].mean - best_st.mean) < 1e-6
                            }
                            continue

                        # Check for tie at the top (not significantly superior to each other)
                        if abs(best_st.mean - runner_up_st.mean) < 1e-6:
                            tied_map = {
                                c[0]: not_significant_color
                                for c in candidates
                                if abs(c[1].mean - best_st.mean) < 1e-6
                            }
                            best_in_col[(eps, loss_type, metric_name)] = tied_map
                            if ttest_records is not None:
                                ttest_records.append({
                                    "explain_method": explain_method,
                                    "model": model_raw,
                                    "eps": eps,
                                    "loss_type": loss_type,
                                    "metric": metric_name,
                                    "best_method": best_m_id,
                                    "best_mean": best_st.mean,
                                    "best_std": best_st.std,
                                    "runner_up_method": runner_up_m_id,
                                    "runner_up_mean": runner_up_st.mean,
                                    "runner_up_std": runner_up_st.std,
                                    "t_stat": 0.0,
                                    "p_value": 1.0,
                                    "alpha": ttest_alpha,
                                    "significant": False,
                                    "color": not_significant_color,
                                    "note": "tie",
                                })
                            continue

                        # Check if both have enough data (valid mean and sd)
                        # "cái nào có đủ sd mean rồi thì t-test với N=5, còn chưa thì thôi"
                        has_enough_sd = (
                            best_st.std is not None
                            and not math.isnan(best_st.std)
                            and runner_up_st.std is not None
                            and not math.isnan(runner_up_st.std)
                            and (best_st.count >= 2 or best_st.std > 0)
                            and (runner_up_st.count >= 2 or runner_up_st.std > 0)
                        )

                        if use_ttest and has_enough_sd:
                            n1 = ttest_n if ttest_n is not None else 5
                            n2 = ttest_n if ttest_n is not None else 5

                            if metric_name == "sro":
                                t_stat, p_val = _run_ttest(
                                    mean1=runner_up_st.mean,
                                    std1=runner_up_st.std,
                                    n1=n2,
                                    mean2=best_st.mean,
                                    std2=best_st.std,
                                    n2=n1,
                                    equal_var=False,
                                    alternative=ttest_alternative,
                                )
                            else:
                                t_stat, p_val = _run_ttest(
                                    mean1=best_st.mean,
                                    std1=best_st.std,
                                    n1=n1,
                                    mean2=runner_up_st.mean,
                                    std2=runner_up_st.std,
                                    n2=n2,
                                    equal_var=False,
                                    alternative=ttest_alternative,
                                )

                            if p_val is not None:
                                is_sig = (p_val < ttest_alpha)
                                col = significant_color if is_sig else not_significant_color
                                best_in_col[(eps, loss_type, metric_name)] = {best_m_id: col}
                                if ttest_records is not None:
                                    ttest_records.append({
                                        "explain_method": explain_method,
                                        "model": model_raw,
                                        "eps": eps,
                                        "loss_type": loss_type,
                                        "metric": metric_name,
                                        "best_method": best_m_id,
                                        "best_mean": best_st.mean,
                                        "best_std": best_st.std,
                                        "runner_up_method": runner_up_m_id,
                                        "runner_up_mean": runner_up_st.mean,
                                        "runner_up_std": runner_up_st.std,
                                        "t_stat": t_stat,
                                        "p_value": p_val,
                                        "alpha": ttest_alpha,
                                        "significant": is_sig,
                                        "color": col,
                                        "note": "welch_ttest",
                                    })
                            else:
                                best_in_col[(eps, loss_type, metric_name)] = {
                                    best_m_id: significant_color
                                }
                        else:
                            # Not enough data for t-test ("còn chưa thì thôi") -> default to significant color (blue)
                            best_in_col[(eps, loss_type, metric_name)] = {
                                best_m_id: significant_color
                            }

        n_rows = len(methods_to_use)
        lines: List[str] = []
        lines.append(f"%{'%' * 71}")
        lines.append(f"\\multirow{{{n_rows}}}{{*}}{{{model_latex}}}")
        lines.append("")

        for idx, (m_id, m_label) in enumerate(methods_to_use):
            cells: List[str] = []
            for eps in [20, 50, 100]:
                for loss_type in ["margin_loss", "negative_cross_entropy_saliency"]:
                    cell_data = (
                        data_for_model.get(m_id, {})
                        .get(str(eps), {})
                        .get(loss_type, {})
                    )
                    asr_st = cell_data.get("asr")
                    sro_st = cell_data.get("sro")
                    h_st = cell_data.get("h_score")

                    if asr_st is None or not isinstance(asr_st, MetricStat):
                        asr_st = MetricStat(float("nan"), float("nan"), 0, [])
                    if sro_st is None or not isinstance(sro_st, MetricStat):
                        sro_st = MetricStat(float("nan"), float("nan"), 0, [])
                    if h_st is None or not isinstance(h_st, MetricStat):
                        h_st = MetricStat(float("nan"), float("nan"), 0, [])

                    asr_color = best_in_col.get((eps, loss_type, "asr"), {}).get(m_id)
                    sro_color = best_in_col.get((eps, loss_type, "sro"), {}).get(m_id)
                    h_color = best_in_col.get((eps, loss_type, "h_score"), {}).get(m_id)

                    cells.append(_format_cell(
                        asr_st,
                        cell_format=cell_format,
                        precision=precision,
                        empty_cell=empty_cell,
                        is_asr=True,
                        asr_percentage=asr_percentage,
                        is_best=(asr_color is not None),
                        highlight_best=highlight_best,
                        best_color=asr_color or significant_color,
                        best_format=best_format,
                    ))
                    cells.append(_format_cell(
                        sro_st,
                        cell_format=cell_format,
                        precision=precision,
                        empty_cell=empty_cell,
                        is_asr=False,
                        is_best=(sro_color is not None),
                        highlight_best=highlight_best,
                        best_color=sro_color or significant_color,
                        best_format=best_format,
                    ))
                    cells.append(_format_cell(
                        h_st,
                        cell_format=cell_format,
                        precision=precision,
                        empty_cell=empty_cell,
                        is_asr=False,
                        is_best=(h_color is not None),
                        highlight_best=highlight_best,
                        best_color=h_color or significant_color,
                        best_format=best_format,
                    ))

            lines.append(f"& {m_label}")
            lines.append("& " + " & ".join(cells) + " \\\\")

        model_blocks.append("\n".join(lines))

    separator = "\n\n\\midrule\n\n"
    content = separator.join(model_blocks)

    table_footer = """
\\bottomrule
\\end{tabular}
}
\\end{table*}"""

    return table_header + "\n\n" + content + "\n\n" + table_footer.strip() + "\n"


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

        margin_h = _h_score(margin.asr, margin.spearman)
        ce_h = _h_score(ce.asr, ce.spearman)

        strategy_text = "Saliency-guided" if strategy == "saliency_guided" else "Uniform"
        row = (
            f"& {eps} & {strategy_text} "
            f"& {margin.asr:.4f} & {margin.spearman:.4f} & {margin_h:.4f} "
            f"& {ce.asr:.4f} & {ce.spearman:.4f} & {ce_h:.4f} \\\\" 
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
                    "objective_curve": _curve_mean([r.objective_curve for r in margin_runs]),
                    "final_asr": sum(r.asr for r in margin_runs) / len(margin_runs),
                },
                "negative_cross_entropy_saliency": {
                    "asr_curve": _curve_mean([r.asr_curve for r in ce_runs]),
                    "saliency_curve": _curve_mean([r.saliency_curve for r in ce_runs]),
                    "objective_curve": _curve_mean([r.objective_curve for r in ce_runs]),
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
    margin_obj_curves = [item["margin_loss"]["objective_curve"] for item in loss_pairs]
    ce_obj_curves = [item["negative_cross_entropy_saliency"]["objective_curve"] for item in loss_pairs]

    margin_final_asr = [float(item["margin_loss"]["final_asr"]) for item in loss_pairs]
    ce_final_asr = [float(item["negative_cross_entropy_saliency"]["final_asr"]) for item in loss_pairs]

    return {
        "num_pairs": len(loss_pairs),
        "averaged_over": "all_models_all_eps_all_strategies_after_filters",
        "margin_loss": {
            "asr_curve": _curve_mean(margin_asr_curves),
            "saliency_curve": _curve_mean(margin_sal_curves),
            "objective_curve": _curve_mean(margin_obj_curves),
            "final_asr": _safe_mean(margin_final_asr),
        },
        "negative_cross_entropy_saliency": {
            "asr_curve": _curve_mean(ce_asr_curves),
            "saliency_curve": _curve_mean(ce_sal_curves),
            "objective_curve": _curve_mean(ce_obj_curves),
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


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _save_text_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def _runstats_from_dict(payload: Dict[str, object]) -> RunStats:
    asr = float(payload.get("asr", 0.0))
    spearman = float(payload.get("spearman", float("nan")))
    raw_h = payload.get("h_score")
    h_score = float(raw_h) if raw_h is not None and not math.isnan(float(raw_h)) else _h_score(asr, spearman)
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
        asr=asr,
        spearman=spearman,
        spearman_failed_samples=int(payload.get("spearman_failed_samples", 0)),
        num_samples_total=int(payload.get("num_samples_total", 0)),
        num_samples_ok=int(payload.get("num_samples_ok", 0)),
        asr_curve=[float(v) for v in payload.get("asr_curve", [])],
        saliency_curve=[float(v) for v in payload.get("saliency_curve", [])],
        objective_curve=[float(v) for v in payload.get("objective_curve", [])],
        seed=str(payload.get("seed", "default")),
        h_score=h_score,
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
        common_title = (
            f"{item['model']} | eps={item['eps']} | {item['strategy']} | "
            f"{item['explain_method']} | {item['algorithm']}"
        )
        base_name = (
            f"{item['model']}__eps-{item['eps']}__strategy-{item['strategy']}"
            f"__exp-{item['explain_method']}__algo-{item['algorithm']}"
        )

        margin_asr = item["margin_loss"]["asr_curve"]
        ce_asr = item["negative_cross_entropy_saliency"]["asr_curve"]
        margin_sal = item["margin_loss"]["saliency_curve"]
        ce_sal = item["negative_cross_entropy_saliency"]["saliency_curve"]
        margin_obj = item["margin_loss"].get("objective_curve", [])
        ce_obj = item["negative_cross_entropy_saliency"].get("objective_curve", [])

        # 1) compare loss: ASR + saliency for both losses.
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

        fig.suptitle(f"{common_title} | compare_loss")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{base_name}__compare-loss.png", dpi=150)
        plt.close(fig)

        # 2) margin objective vs saliency.
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
        ax.plot(range(1, len(margin_obj) + 1), margin_obj, label="margin objective", linewidth=2)
        ax.plot(range(1, len(margin_sal) + 1), margin_sal, label="saliency loss", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.suptitle(f"{common_title} | margin objective vs saliency")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{base_name}__objective-margin-vs-saliency.png", dpi=150)
        plt.close(fig)

        # 3) loglikelihood objective vs saliency.
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
        ax.plot(range(1, len(ce_obj) + 1), ce_obj, label="LogLikeLihood objective", linewidth=2)
        ax.plot(range(1, len(ce_sal) + 1), ce_sal, label="saliency loss", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.suptitle(f"{common_title} | LogLikeLihood objective vs saliency")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{base_name}__objective-loglikelihood-vs-saliency.png", dpi=150)
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
    margin_obj = overall["margin_loss"].get("objective_curve", [])
    ce_obj = overall["negative_cross_entropy_saliency"].get("objective_curve", [])

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

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
    ax.plot(range(1, len(margin_obj) + 1), margin_obj, label="margin objective", linewidth=2.2)
    ax.plot(range(1, len(margin_sal) + 1), margin_sal, label="saliency loss", linewidth=2.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle(f"overall margin objective vs saliency (pairs={overall['num_pairs']})")
    fig.tight_layout()
    fig.savefig(plot_dir / "overall__objective-margin-vs-saliency.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
    ax.plot(range(1, len(ce_obj) + 1), ce_obj, label="LogLikeLihood objective", linewidth=2.2)
    ax.plot(range(1, len(ce_sal) + 1), ce_sal, label="saliency loss", linewidth=2.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle(f"overall LogLikeLihood objective vs saliency (pairs={overall['num_pairs']})")
    fig.tight_layout()
    fig.savefig(plot_dir / "overall__objective-loglikelihood-vs-saliency.png", dpi=170)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score run folders by ASR + Spearman from per-sample summaries, "
            "aggregate metrics across seeds, and generate complete Overleaf LaTeX table."
        )
    )
    parser.add_argument(
        "--result-root",
        "--root",
        dest="result_root",
        type=str,
        default="server_run/server_run",
        help="Root folder with runs (supports multi-tier algo/seed/model/run or flat layout)",
    )
    parser.add_argument("--seed", type=str, default="all", help="Filter by specific seed (e.g. 22520691) or all")
    parser.add_argument("--model", type=str, default="all", help="Filter model name (e.g. resnet18). Use all to keep all")
    parser.add_argument("--all-runs-json", type=str, default=None, help="Optional precomputed all_runs.json path")
    parser.add_argument("--output-dir", type=str, default="evaluate_script/stats_outputs", help="Directory to save output JSON/plots")
    parser.add_argument("--latex-txt", type=str, default="evaluate_script/latex.txt", help="Path to save rendered LaTeX text (full Overleaf table)")
    parser.add_argument("--algorithm", type=str, default="all", help="Filter algorithm (ga, nsgaii, etc.). Use all to keep all")
    parser.add_argument("--explain-method", type=str, default="all", help="Filter explain method. Use all to keep all")
    parser.add_argument("--eps", type=str, default="all", help="Filter epsilon. Use integer value or all")
    parser.add_argument("--w-m", type=str, default="all", help="Filter by w_m. Use numeric value or all")
    parser.add_argument("--w-c", type=str, default="all", help="Backward-compatible alias of --w-s")
    parser.add_argument("--w-s", type=str, default="all", help="Filter by w_s. Use numeric value or all")
    parser.add_argument("--weight-pairs", type=str, default=None, help="Comma-separated wm:ws pairs, e.g. 1:0,0.5:0.5")
    parser.add_argument(
        "--compare-uniform-loss-only",
        action="store_true",
        help=(
            "Deprecated: uniform-only mode is now always enabled. "
            "This flag is kept for backward compatibility."
        ),
    )
    parser.add_argument(
        "--latex-cell-format",
        type=str,
        default="subscript",
        choices=["subscript", "subscript_raw", "pm", "mean_only"],
        help="Format of mean and std in LaTeX table: subscript ($mean_{\\pm std}$), subscript_raw ($mean_{std}$), pm ($mean \\pm std$), or mean_only",
    )
    parser.add_argument(
        "--cached",
        "--load-cached",
        dest="load_cached",
        action="store_true",
        help="Load precomputed runs from <output-dir>/all_runs.json (or specify file with --all-runs-json) without rescanning root folder",
    )
    parser.add_argument(
        "--min-sample-ratio",
        type=float,
        default=0.5,
        help="Minimum ratio of successful samples required per run (default: 0.5, skips runs missing > 50%% samples)",
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=None,
        help="Expected number of samples per run (default: auto-detected from maximum samples found per model)",
    )
    parser.add_argument(
        "--min-seed-ratio",
        type=float,
        default=0.5,
        help="Minimum ratio of seeds required per cell (default: 0.5, skips cells missing > 50%% seeds)",
    )
    parser.add_argument("--asr-percentage", action="store_true", help="Format ASR as percentage [0, 100] instead of decimal [0, 1]")
    parser.add_argument("--precision", type=int, default=4, help="Decimal precision for LaTeX numbers (default: 4)")
    parser.add_argument("--empty-cell", type=str, default="", help="Placeholder for empty LaTeX cells (default: '', produces clean Overleaf table)")
    parser.add_argument(
        "--template-models",
        action="store_true",
        help="Include all standard models (ResNet18, VGG16, ViT-B-32) in LaTeX table even if some have no runs",
    )
    parser.add_argument("--only-active-methods", action="store_true", help="Only render method rows that have at least one valid result")
    parser.add_argument(
        "--latex-caption",
        type=str,
        default=None,
        help="Custom LaTeX table caption. If omitted, automatically generated per explain method.",
    )
    parser.add_argument(
        "--latex-label",
        type=str,
        default=None,
        help="Custom LaTeX table label prefix. If omitted, automatically generated as tab:ga_vs_nsga_<explain_method>.",
    )
    parser.add_argument(
        "--highlight-best",
        dest="highlight_best",
        action="store_true",
        default=True,
        help="Highlight the best result in each column per model block in bold color (default: True)",
    )
    parser.add_argument(
        "--no-highlight-best",
        dest="highlight_best",
        action="store_false",
        help="Disable highlighting the best result in each column per model block",
    )
    parser.add_argument(
        "--best-color",
        type=str,
        default="blue",
        help="LaTeX color name for best results (default: blue, e.g. blue, cyan, teal)",
    )
    parser.add_argument(
        "--best-format",
        type=str,
        default="textbf_textcolor",
        choices=["textbf_textcolor", "textcolor", "textbf"],
        help="Format for best results: textbf_textcolor (\\textbf{\\textcolor{color}{...}}), textcolor (\\textcolor{color}{...}), or textbf (\\textbf{...})",
    )
    parser.add_argument(
        "--use-ttest",
        dest="use_ttest",
        action="store_true",
        default=True,
        help="Use two-sample t-test (scipy.stats.ttest_ind_from_stats) with N=5 to determine color of best result (blue if p < alpha, red if p >= alpha; default: True)",
    )
    parser.add_argument(
        "--no-ttest",
        dest="use_ttest",
        action="store_false",
        help="Disable t-test and highlight all best results in blue",
    )
    parser.add_argument(
        "--ttest-alpha",
        type=float,
        default=0.05 / 3.0,
        help="Significance level alpha for t-test with Bonferroni correction (default: 0.05 / 3 ≈ 0.01667)",
    )
    parser.add_argument(
        "--ttest-n",
        type=int,
        default=5,
        help="Sample size N for two-sample t-test when sufficient mean and std are present (default: 5)",
    )
    parser.add_argument(
        "--ttest-alternative",
        type=str,
        default="two-sided",
        choices=["two-sided", "greater"],
        help="Alternative hypothesis for t-test (default: two-sided)",
    )
    parser.add_argument(
        "--significant-color",
        type=str,
        default="blue",
        help="LaTeX color name for statistically significant best results (default: blue)",
    )
    parser.add_argument(
        "--not-significant-color",
        type=str,
        default="red",
        help="LaTeX color name for not statistically significant best results (default: red)",
    )
    parser.add_argument(
        "--ttest-metrics",
        nargs="+",
        default=["h_score"],
        help="Metrics to apply t-test significance coloring on (default: h_score). Non-tested metrics (e.g. asr, sro) are highlighted in blue without t-test.",
    )
    parser.add_argument(
        "--print-ttest",
        action="store_true",
        help="Print summary of all t-test comparisons to stdout",
    )
    parser.add_argument("--print-latex", action="store_true", help="Print LaTeX table to stdout")
    parser.add_argument("--make-plots", action="store_true", help="Generate PNG plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir)
    latex_txt_path = Path(args.latex_txt)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_algo = None
    if args.algorithm.lower() != "all":
        target_algo = _normalize_algorithm(args.algorithm)

    target_model = None
    if _normalize_model_name(args.model) != "all":
        target_model = args.model

    target_seed = None
    if str(args.seed).strip().lower() != "all":
        target_seed = str(args.seed).strip()

    target_explain_method = None
    if args.explain_method.lower() != "all":
        target_explain_method = args.explain_method

    target_eps = _parse_optional_int_arg(args.eps, "--eps")

    target_w_m = _parse_optional_float_arg(args.w_m, "--w-m")
    raw_w_s = args.w_s if str(args.w_s).strip().lower() != "all" else args.w_c
    target_w_s = _parse_optional_float_arg(raw_w_s, "--w-s")
    target_pairs = _parse_weight_pairs(args.weight_pairs)

    all_runs_file: Optional[Path] = None
    if args.all_runs_json:
        all_runs_file = Path(args.all_runs_json)
    elif args.load_cached:
        all_runs_file = output_dir / "all_runs.json"
        if not all_runs_file.exists():
            raise FileNotFoundError(
                f"Cached runs file not found at: {all_runs_file}. "
                "Run once without --cached (specifying --root) first to scan and save results."
            )
    elif result_root.is_file() and result_root.suffix.lower() == ".json":
        all_runs_file = result_root

    if all_runs_file is not None:
        if not all_runs_file.exists():
            raise FileNotFoundError(f"all_runs JSON not found: {all_runs_file}")
        print(f"Loading precomputed runs from: {all_runs_file} (no rescan needed)")
        all_runs = _load_runs_from_json(all_runs_file)
        if args.min_sample_ratio > 0.0:
            expected_per_model: Dict[str, int] = {}
            for r in all_runs:
                m_norm = _normalize_model_name(r.model)
                expected_per_model[m_norm] = max(expected_per_model.get(m_norm, 0), r.num_samples_ok)
            all_runs = [
                r
                for r in all_runs
                if r.num_samples_ok > 0
                and (
                    r.num_samples_ok
                    >= args.min_sample_ratio
                    * (
                        args.expected_samples
                        if args.expected_samples is not None
                        else expected_per_model.get(_normalize_model_name(r.model), 0)
                    )
                )
            ]
    else:
        all_runs = _load_all_runs(
            result_root,
            target_model=target_model,
            target_eps=target_eps,
            target_algorithm=target_algo,
            target_seed=target_seed,
            target_explain_method=target_explain_method,
            target_w_m=target_w_m,
            target_w_s=target_w_s,
            target_pairs=target_pairs,
            min_sample_ratio=args.min_sample_ratio,
            expected_samples=args.expected_samples,
        )
    
    if not all_runs:
        raise ValueError(f"No valid run found under: {result_root}")

    filtered = all_runs
    if target_model is not None:
        filtered = [r for r in filtered if _normalize_model_name(r.model) == _normalize_model_name(target_model)]
    if target_eps is not None:
        filtered = [r for r in filtered if r.eps == target_eps]
    if target_algo is not None:
        filtered = [r for r in filtered if r.algorithm == target_algo]
    if target_seed is not None:
        filtered = [r for r in filtered if r.seed == target_seed]
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

    # Always ignore saliency-guided runs in aggregation/printing.
    filtered = [r for r in filtered if r.strategy == "uniform"]

    # For wm=0 and ws=1, both fitness formulations are equivalent; mirror missing side
    # so compare-loss outputs remain complete even if only one fitness was executed.
    filtered = _expand_equivalent_losses_for_wm0_ws1(filtered)

    if not filtered:
        raise ValueError("No run left after filtering")

    # Group runs by explain_method so each explanation method gets its own table & stats
    runs_by_explain: Dict[str, List[RunStats]] = {}
    for r in filtered:
        runs_by_explain.setdefault(r.explain_method, []).append(r)

    serialized_all_agg: Dict[str, object] = {}
    overleaf_tables: Dict[str, str] = {}
    combined_sections: List[str] = []
    all_ttest_records: List[Dict[str, object]] = []
    selected_models = [target_model] if target_model is not None else None

    for exp_method in sorted(runs_by_explain.keys()):
        exp_runs = runs_by_explain[exp_method]
        exp_latex_name = _format_explain_latex_name(exp_method)
        clean_tag = _clean_explain_method_tag(exp_method)

        # Aggregate stats across seeds (skipping cells missing > 50% seeds)
        exp_agg = _aggregate_seed_stats(exp_runs, min_seed_ratio=args.min_seed_ratio)
        serialized_all_agg[exp_method] = _serialize_aggregated_stats(exp_agg)

        # Caption & Label
        if args.latex_caption:
            if "{explain_method}" in args.latex_caption:
                caption = args.latex_caption.format(explain_method=exp_latex_name)
            elif "Simple Grad." in args.latex_caption:
                caption = args.latex_caption.replace("Simple Grad.", exp_latex_name)
            else:
                caption = f"{args.latex_caption} ({exp_latex_name})"
        else:
            caption = (
                rf"Comparison between GA with different $\lambda$ values and NSGA-II ({exp_latex_name}). "
                r"$|P|=50$, $N=200$, $p_c=0.4$, $p_m=0.1$, 100 samples. "
                r"Harmonic Score$=\frac{2\,\mathrm{ASR}(1-\mathrm{SRO})}{\mathrm{ASR}+(1-\mathrm{SRO})}$."
            )

        if args.latex_label:
            label = f"{args.latex_label}_{clean_tag}" if len(runs_by_explain) > 1 else args.latex_label
        else:
            label = f"tab:ga_vs_nsga_{clean_tag}"

        table_str = _build_overleaf_table(
            aggregated_stats=exp_agg,
            models=selected_models,
            explain_method=exp_method,
            caption=caption,
            label=label,
            cell_format=args.latex_cell_format,
            precision=args.precision,
            empty_cell=args.empty_cell,
            asr_percentage=args.asr_percentage,
            only_active_methods=args.only_active_methods,
            include_template_models=args.template_models,
            highlight_best=args.highlight_best,
            best_color=args.best_color,
            best_format=args.best_format,
            use_ttest=args.use_ttest,
            ttest_metrics=args.ttest_metrics,
            ttest_alpha=args.ttest_alpha,
            ttest_n=args.ttest_n,
            ttest_alternative=args.ttest_alternative,
            significant_color=args.significant_color,
            not_significant_color=args.not_significant_color,
            ttest_records=all_ttest_records,
        )

        overleaf_tables[exp_method] = table_str

        # Save individual method LaTeX files
        method_txt_path = latex_txt_path.parent / f"latex_{clean_tag}.txt"
        method_tex_path = output_dir / f"overleaf_table_{clean_tag}.tex"
        with open(method_txt_path, "w", encoding="utf-8") as f:
            f.write(table_str)
        with open(method_tex_path, "w", encoding="utf-8") as f:
            f.write(table_str)

        banner = (
            f"%{'%' * 71}\n"
            f"% Explanation Method: {exp_latex_name} ({exp_method})\n"
            f"%{'%' * 71}\n"
        )
        combined_sections.append(banner + table_str)

    # Save aggregated stats JSON (split by explain method)
    _save_json(output_dir / "seed_aggregated_stats.json", serialized_all_agg)

    combined_tables_str = "\n\n\\clearpage\n\n".join(combined_sections) + "\n"

    # Legacy pair curves & rows for plots and backwards compatibility
    loss_pairs = _build_pair_curves_loss(filtered)
    overall_compare_loss = _build_overall_compare_loss(loss_pairs)
    latex_rows = _build_latex_rows(filtered)

    _save_json(output_dir / "all_runs.json", [r.__dict__ | {"run_dir": str(r.run_dir)} for r in filtered])
    _save_json(output_dir / "compare_loss_curves.json", loss_pairs)
    _save_json(output_dir / "compare_loss_curves_overall.json", overall_compare_loss)
    _save_json(output_dir / "latex_rows.json", latex_rows)
    if all_ttest_records:
        _save_json(output_dir / "ttest_results.json", all_ttest_records)

    # Save Overleaf LaTeX tables
    latex_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latex_txt_path, "w", encoding="utf-8") as f:
        f.write(combined_tables_str)
    with open(output_dir / "overleaf_table.tex", "w", encoding="utf-8") as f:
        f.write(combined_tables_str)

    if args.make_plots:
        _plot_pair_curves_loss(loss_pairs, output_dir)
        _plot_overall_compare_loss(overall_compare_loss, output_dir)

    seeds_found = sorted({r.seed for r in filtered})
    models_found = sorted({r.model for r in filtered})
    algos_found = sorted({r.algorithm for r in filtered})

    print(f"Loaded runs: {len(all_runs)}")
    print(f"Runs after filter: {len(filtered)}")
    print(f"Seeds found ({len(seeds_found)}): {', '.join(seeds_found)}")
    print(f"Models: {', '.join(models_found)}")
    print(f"Algorithms: {', '.join(algos_found)}")
    print(f"Explanation methods ({len(runs_by_explain)}): {', '.join(sorted(runs_by_explain.keys()))}")
    print(f"Loss pairs: {len(loss_pairs)}")
    print("Mode: uniform_only (saliency_guided excluded)")
    print(f"Output dir: {output_dir}")
    print(f"Overleaf LaTeX master report: {latex_txt_path}")
    for exp_m in sorted(runs_by_explain.keys()):
        c_tag = _clean_explain_method_tag(exp_m)
        print(f"  - [{exp_m}] -> {latex_txt_path.parent / f'latex_{c_tag}.txt'}")

    if all_ttest_records:
        sig_cnt = sum(1 for r in all_ttest_records if r.get("significant"))
        not_sig_cnt = len(all_ttest_records) - sig_cnt
        print(f"[T-Test] Computed {len(all_ttest_records)} comparisons (N={args.ttest_n}, alpha={args.ttest_alpha:.5f}): {sig_cnt} significant (blue), {not_sig_cnt} non-significant (red)")
        print(f"  -> Saved details to {output_dir / 'ttest_results.json'}")

        if args.print_ttest:
            print("\n" + "=" * 105)
            print(f"{'T-TEST SIGNIFICANCE REPORT (N=' + str(args.ttest_n) + ', alpha=' + f'{args.ttest_alpha:.5f}' + ')':^105}")
            print("=" * 105)
            print(f"{'Explain':<17} {'Model':<10} {'Eps':<4} {'Loss':<10} {'Metric':<7} {'Best (Mean±SD)':<24} {'Runner-up (Mean±SD)':<24} {'p-value':<10} {'Sig?':<5}")
            print("-" * 105)
            for rec in all_ttest_records:
                exp = str(rec.get("explain_method") or "")[:16]
                mod = str(rec.get("model") or "")[:9]
                eps_val = str(rec.get("eps") or "")
                loss_short = "margin" if "margin" in str(rec.get("loss_type")) else "loglik"
                metric_val = str(rec.get("metric") or "")
                bm = rec.get("best_method", "")
                b_mean = rec.get("best_mean", float("nan"))
                b_std = rec.get("best_std", float("nan"))
                rm = rec.get("runner_up_method", "")
                r_mean = rec.get("runner_up_mean", float("nan"))
                r_std = rec.get("runner_up_std", float("nan"))
                pval = rec.get("p_value")
                pval_str = f"{pval:.4e}" if (pval is not None and not math.isnan(pval)) else "N/A"
                sig = "YES" if rec.get("significant") else "NO"
                b_str = f"{bm}:{b_mean:.3f}" + (f"±{b_std:.3f}" if (b_std is not None and not math.isnan(b_std)) else "")
                r_str = f"{rm}:{r_mean:.3f}" + (f"±{r_std:.3f}" if (r_std is not None and not math.isnan(r_std)) else "")
                print(f"{exp:<17} {mod:<10} {eps_val:<4} {loss_short:<10} {metric_val:<7} {b_str:<24} {r_str:<24} {pval_str:<10} {sig:<5}")
            print("=" * 105 + "\n")

    if args.print_latex:
        print("\n=== Overleaf LaTeX Tables ===")
        print(combined_tables_str)

    if args.print_latex:
        print("\n=== Overleaf LaTeX Tables ===")
        print(combined_tables_str)


if __name__ == "__main__":
    main()