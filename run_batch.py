import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image
from tqdm.auto import tqdm



ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(ROOT_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from LossFunctions import MarginSalinecy_Fitness, NegativeCrossEntropySaliency_Fitness
from util import get_explainable_method, get_torchvision_model
from weightedSUM_GA import Weighted_Sum_GA
from NSGAII import NSGAII


DEFAULT_IMAGENET_VAL_ROOT = r"E:\ImageNet1K\imagenet\ImageNet1K\val"
DEFAULT_REMOTE_VAL_ROOT = "/datastore/elo/quanphm/dataset/ImageNet1K/val"
# DEFAULT_REMOTE_VAL_ROOT = DEFAULT_IMAGENET_VAL_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch attack runner based on run_sample.py (no subprocess calls)"
    )
    parser.add_argument(
        "--selection-file",
        type=str,
        default=None,
        help="Path to *_selection.json file. Default: model_evaluation_results/<model_name>_selection.json",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name override. If omitted, inferred from selection filename",
    )
    parser.add_argument(
        "--num_sample",
        "--num-sample",
        dest="num_sample",
        type=int,
        default=None,
        help="Number of samples to run from selection file",
    )
    parser.add_argument(
        "--imagenet-val-root",
        type=str,
        default=DEFAULT_IMAGENET_VAL_ROOT,
        help="Local ImageNet val root folder",
    )
    parser.add_argument(
        "--replace-from-root",
        type=str,
        default=DEFAULT_REMOTE_VAL_ROOT,
        help="Old root path in selection json to be replaced at runtime",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="batch_outputs",
        help="Root folder for all artifacts",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite existing outputs. If omitted, existing entries are skipped",
    )

    parser.add_argument(
        "--explain-method",
        type=str,
        default="simple_gradient",
        choices=["simple_gradient", "integrated_gradients", 'input_gradient'],
        help="Saliency explanation method",
    )
    parser.add_argument("--label", type=int, default=None, help="True label index override")
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--eps", type=int, default=50)
    parser.add_argument("--p-size", type=float, default=1.0)
    parser.add_argument("--pc", type=float, default=0.4)
    parser.add_argument("--pm", type=float, default=0.1)
    parser.add_argument("--zero-probability", type=float, default=0.3)
    parser.add_argument("--w-margin", type=float, default=0.5)
    parser.add_argument("--w-saliency", type=float, default=0.5)
    parser.add_argument(
        "--fitness-function",
        type=str,
        default="margin_saliency",
        choices=["margin_saliency", "negative_cross_entropy_saliency", "cross_entropy_saliency"],
        help="Fitness function to optimize",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="weighted_sum_ga",
        choices=["weighted_sum_ga", "nsgaii"],
        help="Optimization algorithm to run",
    )
    parser.add_argument(
        "--operator-strategy",
        type=str,
        default="uniform",
        choices=["uniform", "saliency_guided"],
    )
    parser.add_argument("--saliency-temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None, help="Global random seed for reproducibility")

    return parser.parse_args()


def _save_saliency_map(saliency_map, output_path):
    map_2d = saliency_map.detach().float().cpu().squeeze()
    map_min = map_2d.min()
    map_max = map_2d.max()
    den = (map_max - map_min).item()

    if den > 1e-12:
        map_2d = (map_2d - map_min) / (map_max - map_min)
    else:
        map_2d = torch.zeros_like(map_2d)

    r = torch.clamp(3.0 * map_2d, 0.0, 1.0)
    g = torch.clamp(3.0 * map_2d - 1.0, 0.0, 1.0)
    b = torch.clamp(3.0 * map_2d - 2.0, 0.0, 1.0)
    rgb = (torch.stack([r, g, b], dim=-1) * 255.0).clamp(0, 255).byte().numpy()
    Image.fromarray(rgb, mode="RGB").save(output_path)


def load_selection_file(selection_file):
    with open(selection_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Selection file must contain a JSON object: {selection_file}")

    return data


def infer_model_name_from_selection_file(selection_file):
    stem = Path(selection_file).stem
    suffix = "_selection"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def resolve_image_path(raw_path, class_name, imagenet_val_root, replace_from_root):
    image_path = Path(raw_path)
    if image_path.exists():
        return image_path

    normalized_raw = str(raw_path).replace("\\", "/")
    normalized_old_root = replace_from_root.replace("\\", "/").rstrip("/")

    if normalized_raw.startswith(normalized_old_root + "/"):
        suffix = normalized_raw[len(normalized_old_root) + 1 :]
        candidate = Path(imagenet_val_root) / Path(suffix)
        if candidate.exists():
            return candidate

    image_name = Path(raw_path).name
    fallback = Path(imagenet_val_root) / class_name / image_name
    return fallback


def prepare_output_paths(output_dir):
    return {
        "adv": output_dir / "adv.png",
        "clean": output_dir / "clean.png",
        "clean_map": output_dir / "clean_map.png",
        "adv_map": output_dir / "adv_map.png",
        "history_txt": output_dir / "history_scores.txt",
        "non_dominated_front_txt": output_dir / "non_dominated_front_scores.txt",
        "non_dominated_front_history_dir": output_dir / "non_dominated_front_history",
        "non_dominated_front_items_dir": output_dir / "non_dominated_front_items",
        "summary": output_dir / "summary.json",
    }


def _to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def save_history_scores_txt(history, output_path):
    # One line per iteration: "margin_loss saliency_loss"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in history:
            margin = _to_float(item["margin_loss"])
            saliency = _to_float(item["saliency_loss"])
            f.write(f"{margin:.12g} {saliency:.12g}\n")


def save_non_dominated_front_txt(front_fitness, output_path):
    if front_fitness is None:
        return
    with open(output_path, "w", encoding="utf-8") as f:
        for row in front_fitness:
            score_1 = float(row[0])
            score_2 = float(row[1])
            f.write(f"{score_1:.12g} {score_2:.12g}\n")


def save_non_dominated_front_history(front_history, output_dir):
    if not front_history:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for it, front_fitness in enumerate(front_history):
        output_path = output_dir / f"iter_{it:04d}.txt"
        save_non_dominated_front_txt(front_fitness, str(output_path))


def save_non_dominated_front_items(front_fitness, front_adv_images, model, normalize, y_true, explain_fn, device, output_dir):
    if front_fitness is None or front_adv_images is None or len(front_adv_images) == 0:
        return None

    if len(front_adv_images) != len(front_fitness):
        raise ValueError(
            f"non-dominated front mismatch: {len(front_adv_images)} images vs {len(front_fitness)} fitness rows"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    adv_batch = torch.stack([img.detach().to(device) for img in front_adv_images], dim=0)
    front_saliency_maps, _ = explain_fn(model, adv_batch, normalize, y_true)

    for line_idx, (adv_img, saliency_map) in enumerate(zip(front_adv_images, front_saliency_maps)):
        adv_path = output_dir / f"line_{line_idx:04d}_adv.png"
        map_path = output_dir / f"line_{line_idx:04d}_map.png"
        save_image(adv_img.detach().cpu(), str(adv_path))
        _save_saliency_map(saliency_map, str(map_path))

    return output_dir


def history_to_lists(history):
    margin = []
    saliency = []
    for item in history:
        margin.append(_to_float(item["margin_loss"]))
        saliency.append(_to_float(item["saliency_loss"]))
    return margin, saliency


def _fmt_num(value):
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.6g}"
        return text.replace("+", "")
    return str(value)


def build_approach_tag(args):
    parts = [
        f"strategy-{args.operator_strategy}",
        f"wm-{_fmt_num(args.w_margin)}",
        f"ws-{_fmt_num(args.w_saliency)}",
        f"eps-{_fmt_num(args.eps)}",
        f"ps-{_fmt_num(args.p_size)}",
        f"zp-{_fmt_num(args.zero_probability)}",
        f"temp-{_fmt_num(args.saliency_temperature)}",
        f"exp-{args.explain_method}",
    ]
    if args.algorithm != "weighted_sum_ga":
        parts.append(f"algo-{args.algorithm}")
    if args.fitness_function != "margin_saliency":
        fit_name = args.fitness_function
        if fit_name == "cross_entropy_saliency":
            fit_name = "negative_cross_entropy_saliency"
        parts.append(f"fit-{fit_name}")
    if args.seed is not None:
        parts.append(f"seed-{args.seed}")
    return "__".join(parts)


def create_attacker(ga_params, algorithm):
    if algorithm == "weighted_sum_ga":
        return Weighted_Sum_GA(ga_params)
    if algorithm == "nsgaii":
        return NSGAII(ga_params)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def create_fitness(fitness_function, model, x_tensor, normalize, y_true, explain_fn):
    if fitness_function == "margin_saliency":
        return MarginSalinecy_Fitness(
            model=model,
            x_tensor=x_tensor,
            normalize=normalize,
            y_true=y_true,
            explain_method=explain_fn,
        )
    if fitness_function in {"negative_cross_entropy_saliency", "cross_entropy_saliency"}:
        return NegativeCrossEntropySaliency_Fitness(
            model=model,
            x_tensor=x_tensor,
            normalize=normalize,
            y_true=y_true,
            explain_method=explain_fn,
        )
    raise ValueError(f"Unsupported fitness function: {fitness_function}")


def run_attack_one(image_path, output_paths, model_name, model, spatial, normalize, explain_fn, args, device, sample_seed=None):
    if sample_seed is not None:
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)

    image = Image.open(image_path).convert("RGB")
    x_tensor = spatial(image).to(device).unsqueeze(0)

    with torch.no_grad():
        pred = model(normalize(x_tensor)).argmax(dim=1)

    y_true = pred if args.label is None else torch.tensor([args.label], device=device)

    fitness = create_fitness(
        fitness_function=args.fitness_function,
        model=model,
        x_tensor=x_tensor,
        normalize=normalize,
        y_true=y_true,
        explain_fn=explain_fn,
    )

    ga_params = {
        "x_tensor": x_tensor,
        "normalize": normalize,
        "fitness": fitness,
        "pop_size": args.pop_size,
        "iterations": args.iterations,
        "eps": args.eps,
        "p_size": args.p_size,
        "pc": args.pc,
        "pm": args.pm,
        "zero_probability": args.zero_probability,
        "all_pixels": torch.arange(x_tensor.shape[-2] * x_tensor.shape[-1], device=device),
        "w_margin": args.w_margin,
        "w_saliency": args.w_saliency,
        "operator_strategy": args.operator_strategy,
        "saliency_temperature": args.saliency_temperature,
        "device": args.device,
    }

    attacker = create_attacker(ga_params, args.algorithm)
    attack_output = attacker.attack()
    non_nominated_front_advimg = None
    if len(attack_output) >= 7:
        (
            adv_chw,
            best_candidate,
            best_scores,
            history,
            non_nominated_front_fitness,
            non_nominated_front_history,
            non_nominated_front_advimg,
        ) = attack_output[:7]
    elif len(attack_output) == 6:
        adv_chw, best_candidate, best_scores, history, non_nominated_front_fitness, non_nominated_front_history = attack_output
    elif len(attack_output) == 5:
        adv_chw, best_candidate, best_scores, history, non_nominated_front_fitness = attack_output
        non_nominated_front_history = None
    else:
        adv_chw, best_candidate, best_scores, history = attack_output
        non_nominated_front_fitness = None
        non_nominated_front_history = None
    adv_chw_cpu = adv_chw.detach().cpu()

    save_image(x_tensor[0].detach().cpu(), str(output_paths["clean"]))
    save_image(adv_chw_cpu, str(output_paths["adv"]))

    with torch.no_grad():
        adv_pred = model(normalize(adv_chw.unsqueeze(0).to(device))).argmax(dim=1).item()

    clean_saliency_map = fitness.saliency_true[0]
    adv_saliency_map, _ = explain_fn(model, adv_chw.unsqueeze(0).to(device), normalize, y_true)
    _save_saliency_map(clean_saliency_map, str(output_paths["clean_map"]))
    _save_saliency_map(adv_saliency_map[0], str(output_paths["adv_map"]))

    save_history_scores_txt(history, str(output_paths["history_txt"]))
    save_non_dominated_front_txt(non_nominated_front_fitness, str(output_paths["non_dominated_front_txt"]))
    save_non_dominated_front_history(non_nominated_front_history, output_paths["non_dominated_front_history_dir"])
    saved_non_dominated_front_items_dir = save_non_dominated_front_items(
        non_nominated_front_fitness,
        non_nominated_front_advimg,
        model,
        normalize,
        y_true,
        explain_fn,
        device,
        output_paths["non_dominated_front_items_dir"],
    )
    history_margin, history_saliency = history_to_lists(history)
    weighted_fitness = best_scores.get("weighted_fitness")
    if weighted_fitness is None:
        weighted_fitness = args.w_margin * float(best_scores["margin_loss"]) + args.w_saliency * float(best_scores["saliency_loss"])

    return {
        "model": model_name,
        "seed": sample_seed,
        "true_label": int(y_true.item()),
        "clean_pred": int(pred.item()),
        "adv_pred": int(adv_pred),
        "l0_distance": int(best_candidate.l0_distance(adv_chw_cpu.to(device))),
        "margin_loss": float(best_scores["margin_loss"]),
        "saliency_loss": float(best_scores["saliency_loss"]),
        "weighted_fitness": float(weighted_fitness),
        "first_success_iteration": best_scores.get("first_success_iteration"),
        "algorithm": args.algorithm,
        "fitness_function": args.fitness_function,
        "operator_strategy": args.operator_strategy,
        "saliency_temperature": float(args.saliency_temperature),
        "history_scores_file": str(output_paths["history_txt"]),
        "non_dominated_front_scores_file": str(output_paths["non_dominated_front_txt"]) if non_nominated_front_fitness is not None else None,
        "non_dominated_front_history_dir": str(output_paths["non_dominated_front_history_dir"]) if non_nominated_front_history else None,
        "non_dominated_front_items_dir": str(saved_non_dominated_front_items_dir) if saved_non_dominated_front_items_dir is not None else None,
        "history_margin": history_margin,
        "history_saliency": history_saliency,
    }


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.num_sample is not None and args.num_sample < 0:
        raise ValueError("--num_sample must be >= 0")

    if args.selection_file is None:
        if not args.model_name:
            raise ValueError("Provide --model-name or --selection-file")
        selection_file = Path("model_evaluation_results") / f"{args.model_name}_selection.json"
        # selection_file = Path("model_evaluation_results") / f"{args.model_name}_selection_random50.json"
    else:
        selection_file = Path(args.selection_file)

    if not selection_file.exists():
        raise FileNotFoundError(f"Selection file not found: {selection_file}")

    model_name = args.model_name or infer_model_name_from_selection_file(selection_file)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU")
        args.device = "cpu"

    device = torch.device(args.device)

    print(f"[INFO] selection_file={selection_file}")
    print(f"[INFO] model={model_name}")
    print(f"[INFO] device={args.device}")
    print(f"[INFO] seed={args.seed}")

    model, spatial, normalize = get_torchvision_model(model_name, pretrained=True)
    model = model.to(device)
    model.eval()

    explain_fn = get_explainable_method(args.explain_method)

    selections = load_selection_file(selection_file)
    items = list(selections.items())

    if args.num_sample is not None:
        items = items[: args.num_sample]

    approach_tag = build_approach_tag(args)
    run_root = Path(args.output_root) / model_name / approach_tag
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] approach={approach_tag}")
    print(f"[INFO] output_dir={run_root}")

    all_results = []
    total_ok = 0
    total_failed = 0
    total_skipped = 0
    total_missing = 0

    progress = tqdm(items, total=len(items), desc="Running attacks", unit="img")

    for sample_index, (class_name, raw_path) in enumerate(progress):
        image_path = resolve_image_path(
            raw_path=raw_path,
            class_name=class_name,
            imagenet_val_root=args.imagenet_val_root,
            replace_from_root=args.replace_from_root,
        )

        image_stem = image_path.stem
        output_dir = run_root / class_name / image_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = prepare_output_paths(output_dir)

        if output_paths["summary"].exists() and not args.replace:
            result = {
                "status": "skipped",
                "reason": "exists",
                "model": model_name,
                "class": class_name,
                "input_raw": raw_path,
                "resolved_image": str(image_path),
                "output_dir": str(output_dir),
            }
            all_results.append(result)
            total_skipped += 1
            progress.set_postfix(status="skipped", cls=class_name)
            print(f"[SKIPPED] class={class_name} image={image_path.name}")
            continue

        if not image_path.exists():
            result = {
                "status": "missing_image",
                "model": model_name,
                "class": class_name,
                "input_raw": raw_path,
                "resolved_image": str(image_path),
                "output_dir": str(output_dir),
            }
            with open(output_paths["summary"], "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            all_results.append(result)
            total_missing += 1
            progress.set_postfix(status="missing_image", cls=class_name)
            print(f"[MISSING_IMAGE] class={class_name} image={image_path.name}")
            continue

        try:
            sample_seed = None if args.seed is None else (args.seed + sample_index)
            metrics = run_attack_one(
                image_path=image_path,
                output_paths=output_paths,
                model_name=model_name,
                model=model,
                spatial=spatial,
                normalize=normalize,
                explain_fn=explain_fn,
                args=args,
                device=device,
                sample_seed=sample_seed,
            )
            result = {
                "status": "ok",
                "class": class_name,
                "input_raw": raw_path,
                "resolved_image": str(image_path),
                "output_dir": str(output_dir),
            }
            result.update(metrics)
            total_ok += 1
            progress.set_postfix(status="ok", cls=class_name)
            print(f"[OK] class={class_name} image={image_path.name}")
        except Exception as exc:
            result = {
                "status": "failed",
                "model": model_name,
                "class": class_name,
                "input_raw": raw_path,
                "resolved_image": str(image_path),
                "output_dir": str(output_dir),
                "error": str(exc),
            }
            total_failed += 1
            progress.set_postfix(status="failed", cls=class_name)
            print(f"[FAILED] class={class_name} image={image_path.name} error={exc}")

        with open(output_paths["summary"], "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        all_results.append(result)

    report = {
        "selection_file": str(selection_file),
        "model": model_name,
        "approach": approach_tag,
        "seed": args.seed,
        "num_requested": args.num_sample,
        "total": len(all_results),
        "ok": total_ok,
        "failed": total_failed,
        "missing_image": total_missing,
        "skipped": total_skipped,
        "imagenet_val_root": args.imagenet_val_root,
        "replace_from_root": args.replace_from_root,
        "results": all_results,
    }

    report_path = run_root / "batch_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== Batch summary ===")
    print(f"report: {report_path}")
    print(f"total: {report['total']}")
    print(f"ok: {report['ok']}")
    print(f"failed: {report['failed']}")
    print(f"missing_image: {report['missing_image']}")
    print(f"skipped: {report['skipped']}")


if __name__ == "__main__":
    main()
