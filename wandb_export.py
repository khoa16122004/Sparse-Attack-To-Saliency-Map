#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

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


def summarize_report(report):
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

    return {
        "num_ok": len(ok_items),
        "num_total": len(report.get("results", [])),
        "attack_success_rate": safe_mean(success_flags),
        "mean_margin_loss": safe_mean(margin_losses),
        "mean_saliency_loss": safe_mean(saliency_losses),
        "mean_l0_distance": safe_mean(l0_distances),
    }


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


def export_approach(args, model_name, approach_dir):
    approach = approach_dir.name
    report, source = load_report_with_fallback(approach_dir)
    if report is None:
        return False

    parsed = parse_approach_tag(approach)
    fit = normalize_fit(parsed.get("fit"))
    algo = normalize_algo(parsed.get("algo"))
    strategy = parsed.get("strategy", "unknown")

    run_name = f"{model_name}__{strategy}__{fit}__{algo}"

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
        job_type="export",
        config=config,
        tags=args.tags,
        reinit=True,
        mode=args.mode,
    )

    summary = summarize_report(report)
    for k, v in summary.items():
        if v is not None:
            run.summary[k] = v

    ok_items = [item for item in report.get("results", []) if item.get("status") == "ok"]

    margin_curves = []
    saliency_curves = []
    for item in ok_items:
        margin, saliency = load_item_history(item, approach_dir)
        if margin is not None and saliency is not None:
            margin_curves.append(margin)
            saliency_curves.append(saliency)

    margin_mean = mean_curve(margin_curves)
    saliency_mean = mean_curve(saliency_curves)

    max_len = max(len(margin_mean), len(saliency_mean)) if (margin_mean or saliency_mean) else 0
    for i in range(max_len):
        payload = {"iteration": i}
        if i < len(margin_mean):
            payload["curve/margin_mean"] = margin_mean[i]
        if i < len(saliency_mean):
            payload["curve/saliency_mean"] = saliency_mean[i]
        run.log(payload)

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

    run.finish()
    return True


def main():
    args = parse_args()

    if not args.input_root.exists() or not args.input_root.is_dir():
        raise FileNotFoundError(f"input_root not found: {args.input_root}")

    exported = 0
    model_dirs = sorted([p for p in args.input_root.iterdir() if p.is_dir()])
    for model_dir in model_dirs:
        approach_dirs = sorted([p for p in model_dir.iterdir() if p.is_dir()])
        for approach_dir in approach_dirs:
            ok = export_approach(args, model_dir.name, approach_dir)
            if ok:
                exported += 1
                print(f"[OK] exported: {model_dir.name}/{approach_dir.name}")

    print(f"[DONE] Exported {exported} runs to wandb")


if __name__ == "__main__":
    main()
