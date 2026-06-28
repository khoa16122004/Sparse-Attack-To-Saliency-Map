#!/usr/bin/env python3
import argparse
import hashlib
import json
import time
from pathlib import Path

from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export batch_outputs reports to TensorBoard (summary, curves, sample images)"
    )
    parser.add_argument("--input-root", type=Path, default=Path("compare_loss_50"))
    parser.add_argument("--logdir", type=Path, default=Path("tb_logs"))
    parser.add_argument("--sample-images", type=int, default=4)
    parser.add_argument("--watch", action="store_true", help="Continuously re-export on interval")
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
    margins = []
    saliencies = []
    l0s = []

    for item in ok_items:
        true_label = item.get("true_label")
        adv_pred = item.get("adv_pred")
        if isinstance(true_label, int) and isinstance(adv_pred, int):
            success_flags.append(1 if adv_pred != true_label else 0)

        margins.append(item.get("margin_loss"))
        saliencies.append(item.get("saliency_loss"))
        l0s.append(item.get("l0_distance"))

    return {
        "num_ok": len(ok_items),
        "num_total": len(report.get("results", [])),
        "attack_success_rate": safe_mean(success_flags),
        "mean_margin_loss": safe_mean(margins),
        "mean_saliency_loss": safe_mean(saliencies),
        "mean_l0_distance": safe_mean(l0s),
    }


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


def resolve_output_dir(item, approach_dir):
    output_dir = item.get("output_dir")
    if not output_dir:
        return None

    candidate = Path(output_dir)
    if candidate.is_absolute():
        return candidate

    root_dir = approach_dir.parent.parent
    return root_dir / candidate


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


def image_to_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = pil_to_tensor(image).float() / 255.0
    return tensor


def run_name_for_approach(approach):
    parsed = parse_approach_tag(approach)
    strategy = parsed.get("strategy", "unknown")
    fit = normalize_fit(parsed.get("fit"))
    algo = normalize_algo(parsed.get("algo"))

    digest = hashlib.md5(approach.encode("utf-8")).hexdigest()[:8]
    return f"{strategy}__{fit}__{algo}__{digest}"


def export_once(input_root, logdir, sample_images):
    if not input_root.exists():
        print(f"[WARN] input_root not found: {input_root}")
        return

    logdir.mkdir(parents=True, exist_ok=True)

    index = {}
    model_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])

    for model_dir in model_dirs:
        model_name = model_dir.name
        approach_dirs = sorted([p for p in model_dir.iterdir() if p.is_dir()])

        for approach_dir in approach_dirs:
            approach = approach_dir.name
            report, source = load_report_with_fallback(approach_dir)
            if report is None:
                continue

            run_name = run_name_for_approach(approach)
            run_dir = logdir / model_name / run_name
            writer = SummaryWriter(str(run_dir))

            index_key = f"{model_name}/{run_name}"
            index[index_key] = {
                "model": model_name,
                "approach": approach,
                "source": source,
                "path": str(approach_dir),
            }

            summary = summarize_report(report)
            writer.add_text("meta/model", model_name, 0)
            writer.add_text("meta/approach", approach, 0)
            writer.add_text("meta/report_source", str(source), 0)

            for key, value in summary.items():
                if value is not None:
                    writer.add_scalar(f"summary/{key}", value, 0)

            ok_items = [item for item in report.get("results", []) if item.get("status") == "ok"]
            margin_curves = []
            saliency_curves = []
            margin_values = []
            saliency_values = []
            l0_values = []

            for item in ok_items:
                m = item.get("margin_loss")
                s = item.get("saliency_loss")
                l0 = item.get("l0_distance")
                if isinstance(m, (int, float)):
                    margin_values.append(float(m))
                if isinstance(s, (int, float)):
                    saliency_values.append(float(s))
                if isinstance(l0, (int, float)):
                    l0_values.append(float(l0))

                margin, saliency = load_item_history(item, approach_dir)
                if margin is not None and saliency is not None:
                    margin_curves.append(margin)
                    saliency_curves.append(saliency)

            if margin_values:
                writer.add_histogram("distribution/margin_loss", torch.tensor(margin_values), 0)
            if saliency_values:
                writer.add_histogram("distribution/saliency_loss", torch.tensor(saliency_values), 0)
            if l0_values:
                writer.add_histogram("distribution/l0_distance", torch.tensor(l0_values), 0)

            margin_mean = mean_curve(margin_curves)
            saliency_mean = mean_curve(saliency_curves)
            for i, value in enumerate(margin_mean):
                writer.add_scalar("curve/margin_mean", value, i)
            for i, value in enumerate(saliency_mean):
                writer.add_scalar("curve/saliency_mean", value, i)

            if sample_images > 0:
                sorted_items = sorted(ok_items, key=lambda x: str(x.get("output_dir", "")))
                for idx, item in enumerate(sorted_items[:sample_images]):
                    output_dir = resolve_output_dir(item, approach_dir)
                    if output_dir is None:
                        continue

                    for image_key in ["clean.png", "adv.png", "clean_map.png", "adv_map.png"]:
                        image_path = output_dir / image_key
                        if not image_path.exists():
                            continue
                        try:
                            tensor = image_to_tensor(image_path)
                            writer.add_image(f"samples/{idx:02d}/{image_key[:-4]}", tensor, 0)
                        except Exception:
                            continue

            writer.flush()
            writer.close()

    index_path = logdir / "runs_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Exported TensorBoard logs to: {logdir}")
    print(f"[INFO] Run index: {index_path}")


def main():
    args = parse_args()

    if args.watch and args.interval <= 0:
        raise ValueError("--interval must be > 0")

    if args.watch:
        print("[INFO] Watch mode enabled")
        while True:
            export_once(args.input_root, args.logdir, args.sample_images)
            time.sleep(args.interval)
    else:
        export_once(args.input_root, args.logdir, args.sample_images)


if __name__ == "__main__":
    main()
