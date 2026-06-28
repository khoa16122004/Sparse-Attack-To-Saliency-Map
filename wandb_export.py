#!/usr/bin/env python3
import argparse
import hashlib
import json
import time
from pathlib import Path

from PIL import Image
import wandb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export compare_loss_50 results to Weights & Biases"
    )
    parser.add_argument("--input-root", type=Path, default=Path("compare_loss_50"))
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--group", type=str, default="compare_loss_50")
    parser.add_argument("--sample-images", type=int, default=4)
    parser.add_argument("--mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--tags", nargs="*", default=["compare_loss_50", "sparse_attack"])
    parser.add_argument("--watch", action="store_true", help="Continuously scan folder and sync changed runs")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in seconds")
    return parser.parse_args()


def parse_approach_tag(tag):
    parsed = {}
    for part in tag.split("__"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        parsed[key] = value
    return parsed


def normalize_fit(value):
    if not value:
        return "margin_saliency"
    value = str(value).strip().lower()
    if value in {"margin", "margin_loss", "margin_saliency"}:
        return "margin_saliency"
    if value in {"cross_entropy_saliency", "negative_cross_entropy_saliency"}:
        return "negative_cross_entropy_saliency"
    return value


def normalize_algo(value):
    if not value:
        return "weighted_sum_ga"
    value = str(value).strip().lower()
    if value in {"weighted_sum_ga", "weighted_sum", "ga", "default"}:
        return "weighted_sum_ga"
    if value in {"nsgaii", "nsga2", "nsga_ii"}:
        return "nsgaii"
    return value


def to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rebuild_report_from_summaries(approach_dir):
    summary_files = sorted(approach_dir.glob("*/*/summary.json"))
    if not summary_files:
        return None

    results = []
    for summary_file in summary_files:
        try:
            item = load_json(summary_file)
        except Exception:
            continue
        if not isinstance(item, dict):
            continue

        if "output_dir" not in item:
            item["output_dir"] = str(summary_file.parent)
        if "class" not in item:
            item["class"] = summary_file.parent.parent.name
        results.append(item)

    if not results:
        return None

    return {"results": results, "_generated_from": "summary_files"}


def load_report_with_fallback(approach_dir):
    report_path = approach_dir / "batch_report.json"
    if report_path.exists():
        try:
            loaded = load_json(report_path)
            if isinstance(loaded, dict):
                ok_items = [item for item in loaded.get("results", []) if isinstance(item, dict) and item.get("status") == "ok"]
                if ok_items:
                    return loaded, "batch_report"

                rebuilt = rebuild_report_from_summaries(approach_dir)
                if rebuilt is not None:
                    rebuilt_ok = [item for item in rebuilt.get("results", []) if isinstance(item, dict) and item.get("status") == "ok"]
                    if rebuilt_ok:
                        return rebuilt, "summary_files"
            return loaded, "batch_report"
        except Exception:
            pass

    rebuilt = rebuild_report_from_summaries(approach_dir)
    if rebuilt is not None:
        return rebuilt, "summary_files"

    return None, None


def safe_mean(values):
    valid = [v for v in values if isinstance(v, (int, float))]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _rankdata(values):
    indexed = sorted([(v, i) for i, v in enumerate(values)], key=lambda t: t[0])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        j = idx + 1
        while j < len(indexed) and indexed[j][0] == indexed[idx][0]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        for k in range(idx, j):
            ranks[indexed[k][1]] = avg_rank
        idx = j
    return ranks


def _spearman_rank_corr(x, y):
    if len(x) != len(y):
        return None
    n = len(x)
    if n < 2:
        return None

    rx = _rankdata(x)
    ry = _rankdata(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return None
    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


def _load_gray_flat_image(image_path):
    try:
        img = Image.open(image_path).convert("L")
        return [float(v) for v in img.getdata()]
    except Exception:
        return None


def compute_spearman_adv_vs_clean_map(item, approach_dir):
    output_dir = resolve_output_dir(item, approach_dir)
    if output_dir is None:
        return None

    clean_map_path = output_dir / "clean_map.png"
    adv_map_path = output_dir / "adv_map.png"
    if not clean_map_path.exists() or not adv_map_path.exists():
        return None

    clean_vals = _load_gray_flat_image(clean_map_path)
    adv_vals = _load_gray_flat_image(adv_map_path)
    if clean_vals is None or adv_vals is None:
        return None

    return _spearman_rank_corr(clean_vals, adv_vals)


def summarize_report(report, approach_dir=None):
    ok_items = [item for item in report.get("results", []) if item.get("status") == "ok"]

    success_flags = []
    margin_losses = []
    saliency_losses = []
    l0_distances = []

    for item in ok_items:
        true_label = item.get("true_label")
        adv_pred = item.get("adv_pred")
        if isinstance(true_label, int) and isinstance(adv_pred, int):
            success_flags.append(1 if adv_pred != true_label else 0)

        margin_losses.append(item.get("margin_loss"))
        saliency_losses.append(item.get("saliency_loss"))
        l0_distances.append(item.get("l0_distance"))

    spearman_values = []
    if approach_dir is not None:
        for item in ok_items:
            spearman_values.append(compute_spearman_adv_vs_clean_map(item, approach_dir))

    return {
        "num_ok": len(ok_items),
        "num_total": len(report.get("results", [])),
        "attack_success_rate": safe_mean(success_flags),
        "mean_margin_loss": safe_mean(margin_losses),
        "mean_saliency_loss": safe_mean(saliency_losses),
        "mean_l0_distance": safe_mean(l0_distances),
        "mean_spearman_adv_vs_clean_map": safe_mean(spearman_values),
    }


def build_cumulative_asr_by_iteration(primary_curves, success_threshold=0.0):
    if not primary_curves:
        return []

    max_len = max(len(curve) for curve in primary_curves)
    if max_len <= 0:
        return []

    # success_flags[i] indicates whether sample i has ever succeeded up to current iteration.
    success_flags = [False] * len(primary_curves)
    series = []

    for iteration in range(max_len):
        for idx, curve in enumerate(primary_curves):
            if success_flags[idx]:
                continue
            if iteration < len(curve) and curve[iteration] <= success_threshold:
                success_flags[idx] = True

        success_count = sum(1 for flag in success_flags if flag)
        asr_value = float(success_count / len(primary_curves))
        series.append(
            {
                "iteration": iteration,
                "asr_cumulative": asr_value,
                "success_count": success_count,
                "num_samples": len(primary_curves),
            }
        )

    return series


def resolve_output_dir(item, approach_dir):
    output_dir = item.get("output_dir")
    if not output_dir:
        return None

    candidate = Path(output_dir)
    if candidate.is_absolute():
        return candidate

    root_dir = approach_dir.parent.parent
    return root_dir / candidate


def to_float_list(values):
    if not isinstance(values, list):
        return None
    out = []
    for v in values:
        if not isinstance(v, (int, float)):
            return None
        out.append(float(v))
    return out


def load_history_from_txt(txt_path):
    if not txt_path.exists():
        return None, None

    margin = []
    saliency = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                margin.append(float(parts[0]))
                saliency.append(float(parts[1]))
            except ValueError:
                continue

    if not margin or len(margin) != len(saliency):
        return None, None

    return margin, saliency


def load_item_history(item, approach_dir):
    margin = to_float_list(item.get("history_margin"))
    saliency = to_float_list(item.get("history_saliency"))
    if margin is not None and saliency is not None and len(margin) == len(saliency):
        return margin, saliency

    output_dir = resolve_output_dir(item, approach_dir)
    if output_dir is None:
        return None, None

    return load_history_from_txt(output_dir / "history_scores.txt")


def mean_curve(curves):
    if not curves:
        return []

    max_len = max(len(c) for c in curves)
    out = []
    for idx in range(max_len):
        vals = [c[idx] for c in curves if idx < len(c)]
        if vals:
            out.append(float(sum(vals) / len(vals)))
        else:
            out.append(0.0)
    return out


def build_run_id(model_name, approach):
    raw = f"{model_name}::{approach}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def log_final_summary_table(args, rows):
    if not rows:
        return

    run_id_src = f"{args.project}:{args.group}:final_table".encode("utf-8")
    run_id = hashlib.md5(run_id_src).hexdigest()
    run_name = f"{args.group}__final_metrics_table"

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group,
        name=run_name,
        id=run_id,
        resume="allow",
        job_type="report_table",
        tags=args.tags,
        reinit=True,
        mode=args.mode,
    )

    columns = [
        "model",
        "approach",
        "strategy",
        "eps",
        "fitness",
        "algorithm",
        "num_ok",
        "num_total",
        "asr",
        "mean_margin_loss",
        "mean_saliency_loss",
        "mean_l0_distance",
        "mean_spearman_adv_vs_clean_map",
        "report_source",
    ]

    data = []
    for row in rows:
        data.append([
            row.get("model"),
            row.get("approach"),
            row.get("strategy"),
            row.get("eps"),
            row.get("fitness"),
            row.get("algorithm"),
            row.get("num_ok"),
            row.get("num_total"),
            row.get("attack_success_rate"),
            row.get("mean_margin_loss"),
            row.get("mean_saliency_loss"),
            row.get("mean_l0_distance"),
            row.get("mean_spearman_adv_vs_clean_map"),
            row.get("report_source"),
        ])

    table = wandb.Table(columns=columns, data=data)
    run.log({"report/final_metrics_table": table})
    run.summary["report/num_rows"] = len(rows)
    run.finish()


def compute_approach_signature(approach_dir):
    parts = []

    report_path = approach_dir / "batch_report.json"
    if report_path.exists():
        stat = report_path.stat()
        parts.append(f"report:{stat.st_mtime_ns}:{stat.st_size}")

    summary_files = sorted(approach_dir.glob("*/*/summary.json"))
    parts.append(f"summary_count:{len(summary_files)}")
    for path in summary_files:
        stat = path.stat()
        rel = path.relative_to(approach_dir)
        parts.append(f"{rel}:{stat.st_mtime_ns}:{stat.st_size}")

    if not parts:
        return None

    digest_src = "|".join(parts).encode("utf-8")
    return hashlib.md5(digest_src).hexdigest()


def export_approach(args, model_name, approach_dir, progress_cache=None):
    approach = approach_dir.name
    report, source = load_report_with_fallback(approach_dir)
    if report is None:
        return False

    parsed = parse_approach_tag(approach)
    fit = normalize_fit(parsed.get("fit"))
    algo = normalize_algo(parsed.get("algo"))
    strategy = parsed.get("strategy", "unknown")
    eps = parsed.get("eps", "na")
    wm = to_float(parsed.get("wm"), default=None)
    ws = to_float(parsed.get("ws"), default=None)

    run_name = f"{model_name}__eps-{eps}__{strategy}__{fit}__{algo}"
    run_id = build_run_id(model_name, approach)
    cache_key = f"{model_name}/{approach}"

    config = {
        "model": model_name,
        "approach": approach,
        "strategy": strategy,
        "fitness": fit,
        "algorithm": algo,
        "report_source": source,
        "input_root": str(args.input_root),
    }
    config.update(parsed)

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group,
        name=run_name,
        id=run_id,
        resume="allow",
        job_type="export",
        config=config,
        tags=args.tags,
        reinit=True,
        mode=args.mode,
    )

    run.define_metric("iteration")
    run.define_metric("curve/*", step_metric="iteration")
    run.define_metric("asr/*", step_metric="iteration")
    run.define_metric("progress/*", step_metric="iteration")

    summary = summarize_report(report, approach_dir=approach_dir)
    for k, v in summary.items():
        if v is not None:
            run.summary[k] = v

    ok_items = [item for item in report.get("results", []) if item.get("status") == "ok"]
    run.summary["progress/ok_items"] = len(ok_items)
    run.summary["progress/total_items"] = len(report.get("results", []))

    margin_curves = []
    saliency_curves = []
    weighted_curves = []
    for item in ok_items:
        margin, saliency = load_item_history(item, approach_dir)
        if margin is not None and saliency is not None:
            margin_curves.append(margin)
            saliency_curves.append(saliency)

            if wm is not None and ws is not None:
                weighted_curves.append([
                    wm * m + ws * s for m, s in zip(margin, saliency)
                ])

    margin_mean = mean_curve(margin_curves)
    saliency_mean = mean_curve(saliency_curves)
    weighted_mean = mean_curve(weighted_curves)

    max_len = max(len(margin_mean), len(saliency_mean), len(weighted_mean)) if (margin_mean or saliency_mean or weighted_mean) else 0
    curve_start = 0
    if progress_cache is not None:
        curve_start = int(progress_cache.get(cache_key, {}).get("curve_next_iter", 0))

    for i in range(curve_start, max_len):
        payload = {"iteration": i}
        if i < len(margin_mean):
            payload["curve/margin_mean"] = margin_mean[i]
        if i < len(saliency_mean):
            payload["curve/saliency_mean"] = saliency_mean[i]
        if i < len(weighted_mean):
            payload["curve/weighted_mean"] = weighted_mean[i]
        run.log(payload)

    asr_series = build_cumulative_asr_by_iteration(margin_curves, success_threshold=0.0)
    if asr_series:
        run.summary["asr/final_cumulative"] = asr_series[-1]["asr_cumulative"]
        run.summary["progress/processed_samples"] = asr_series[-1]["num_samples"]
    else:
        run.summary["progress/processed_samples"] = 0

    asr_start = 0
    if progress_cache is not None:
        asr_start = int(progress_cache.get(cache_key, {}).get("asr_next_iter", 0))

    for point in asr_series[asr_start:]:
        run.log(
            {
                "iteration": point["iteration"],
                "asr/cumulative": point["asr_cumulative"],
                "progress/processed_samples": point["num_samples"],
                "progress/success_count": point["success_count"],
            }
        )

    sample_items = sorted(ok_items, key=lambda x: str(x.get("output_dir", "")))[: args.sample_images]
    image_payload = {}
    for idx, item in enumerate(sample_items):
        output_dir = resolve_output_dir(item, approach_dir)
        if output_dir is None:
            continue

        for name in ["clean.png", "adv.png", "clean_map.png", "adv_map.png"]:
            image_path = output_dir / name
            if image_path.exists():
                image_payload[f"sample/{idx:02d}/{name[:-4]}"] = wandb.Image(str(image_path))

    if image_payload:
        run.log(image_payload)

    if progress_cache is not None:
        progress_cache[cache_key] = {
            "curve_next_iter": max_len,
            "asr_next_iter": len(asr_series),
        }

    run.finish()
    return {
        "model": model_name,
        "approach": approach,
        "strategy": strategy,
        "eps": eps,
        "fitness": fit,
        "algorithm": algo,
        "report_source": source,
        **summary,
    }


def export_once(args, last_signatures=None, progress_cache=None):
    exported = 0
    seen = 0
    summary_rows = []
    model_dirs = sorted([p for p in args.input_root.iterdir() if p.is_dir()])
    for model_dir in model_dirs:
        approach_dirs = sorted([p for p in model_dir.iterdir() if p.is_dir()])
        for approach_dir in approach_dirs:
            key = f"{model_dir.name}/{approach_dir.name}"
            signature = compute_approach_signature(approach_dir)
            if signature is None:
                continue

            seen += 1
            if last_signatures is not None:
                prev = last_signatures.get(key)
                if prev == signature:
                    continue

            row = export_approach(args, model_dir.name, approach_dir, progress_cache=progress_cache)
            if row:
                exported += 1
                print(f"[OK] exported: {key}")
                summary_rows.append(row)

            if last_signatures is not None:
                last_signatures[key] = signature

    if summary_rows:
        log_final_summary_table(args, summary_rows)

    return exported, seen


def main():
    args = parse_args()

    if not args.input_root.exists() or not args.input_root.is_dir():
        raise FileNotFoundError(f"input_root not found: {args.input_root}")

    if args.watch and args.interval <= 0:
        raise ValueError("--interval must be > 0")

    if args.watch:
        print(f"[INFO] Watch mode enabled, interval={args.interval}s")
        signatures = {}
        progress_cache = {}
        while True:
            exported, seen = export_once(args, last_signatures=signatures, progress_cache=progress_cache)
            print(f"[INFO] scan done: seen={seen}, updated={exported}")
            time.sleep(args.interval)
    else:
        exported, seen = export_once(args, last_signatures=None, progress_cache=None)
        print(f"[DONE] Exported {exported} runs to wandb (seen={seen})")


if __name__ == "__main__":
    main()
