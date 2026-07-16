import argparse
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(ROOT_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from LossFunctions import MarginSalinecy_Fitness, NegativeCrossEntropySaliency_Fitness
from util import get_explainable_method, get_torchvision_model, save_attack_two_score_charts
from weightedSUM_GA import Weighted_Sum_GA
from NSGAII import NSGAII


def parse_args():
    parser = argparse.ArgumentParser(description="Run sparse attack with weighted-sum genetic algorithm")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="adv.png", help="Path to save adversarial image")
    parser.add_argument("--clean-image-output", type=str, default=None, help="Path to save resized clean image")
    parser.add_argument("--clean-map-output", type=str, default=None, help="Path to save clean/original saliency map")
    parser.add_argument("--adv-map-output", type=str, default=None, help="Path to save adversarial saliency map")
    parser.add_argument("--save-history-chart", action="store_true", help="Save separate charts for margin and saliency scores")
    parser.add_argument("--history-chart-output", type=str, default=None, help="Base path used to derive *_margin and *_saliency chart files")
    parser.add_argument("--margin-chart-output", type=str, default=None, help="Path to save margin score chart")
    parser.add_argument("--saliency-chart-output", type=str, default=None, help="Path to save saliency score chart")
    parser.add_argument("--model", type=str, default="resnet50", help="Torchvision model name")
    parser.add_argument("--label", type=int, default=None, help="True label index. If omitted, uses model prediction")
    parser.add_argument(
        "--explain-method",
        type=str,
        default="simple_gradient",
        choices=["simple_gradient", "integrated_gradients", "input_gradient", "grad_cam"],
        help="Saliency explanation method",
    )

    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--eps", type=int, default=50, help="Number of perturbed pixels")
    parser.add_argument("--p-size", type=float, default=1.0, help="Perturbation step size")
    parser.add_argument("--pc", type=float, default=0.9, help="Crossover ratio")
    parser.add_argument("--pm", type=float, default=0.1, help="Mutation ratio")
    parser.add_argument("--zero-probability", type=float, default=0.3, help="Probability of zero channel perturbation")
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
        help="Population initialization strategy (saliency_guided affects init only)",
    )
    parser.add_argument(
        "--saliency-temperature",
        type=float,
        default=1.0,
        help="Temperature for saliency-guided pixel sampling (lower is sharper)",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None, help="Global random seed for reproducibility")

    return parser.parse_args()


def _save_saliency_map(saliency_map, output_path):
    # Save a single saliency map with a standard "hot" colormap.
    map_2d = saliency_map.detach().float().cpu().squeeze()
    map_min = map_2d.min()
    map_max = map_2d.max()
    den = (map_max - map_min).item()
    if den > 1e-12:
        map_2d = (map_2d - map_min) / (map_max - map_min)
    else:
        map_2d = torch.zeros_like(map_2d)

    # Classic hot map: black -> red -> yellow -> white.
    r = torch.clamp(3.0 * map_2d, 0.0, 1.0)
    g = torch.clamp(3.0 * map_2d - 1.0, 0.0, 1.0)
    b = torch.clamp(3.0 * map_2d - 2.0, 0.0, 1.0)
    rgb = (torch.stack([r, g, b], dim=-1) * 255.0).clamp(0, 255).byte().numpy()
    Image.fromarray(rgb, mode="RGB").save(output_path)


def _save_four_class_maps(model, explain_fn, normalize, clean_chw, adv_chw, class_a, class_b, device, output_paths):
    clean_batch = clean_chw.unsqueeze(0).to(device)
    adv_batch = adv_chw.unsqueeze(0).to(device)

    target_a = torch.tensor([int(class_a)], device=device)
    target_b = torch.tensor([int(class_b)], device=device)

    clean_map_a, _ = explain_fn(model, clean_batch, normalize, target_a)
    clean_map_b, _ = explain_fn(model, clean_batch, normalize, target_b)
    adv_map_a, _ = explain_fn(model, adv_batch, normalize, target_a)
    adv_map_b, _ = explain_fn(model, adv_batch, normalize, target_b)

    _save_saliency_map(clean_map_a[0], output_paths["clean_a"])
    _save_saliency_map(clean_map_b[0], output_paths["clean_b"])
    _save_saliency_map(adv_map_a[0], output_paths["adv_a"])
    _save_saliency_map(adv_map_b[0], output_paths["adv_b"])


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


def _sorted_front_by_score1(front_fitness):
    if front_fitness is None:
        return None

    order = np.argsort(front_fitness[:, 0], kind="mergesort")
    return front_fitness[order]


def save_non_dominated_front_txt(front_fitness, output_path):
    if front_fitness is None:
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in front_fitness:
            score_1 = float(row[0])
            score_2 = float(row[1])
            f.write(f"{score_1:.12g} {score_2:.12g}\n")


def run_attack(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    model, spatial, normalize = get_torchvision_model(args.model, pretrained=True)
    model = model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    x_tensor = spatial(image).to(device).unsqueeze(0)

    with torch.no_grad():
        pred = model(normalize(x_tensor)).argmax(dim=1)
    y_true = pred if args.label is None else torch.tensor([args.label], device=device)
    print("Original PRediction: ", pred)

    explain_fn = get_explainable_method(args.explain_method)

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
    if len(attack_output) >= 5:
        adv_chw, best_candidate, best_scores, history, non_nominated_front_fitness = attack_output
    else:
        adv_chw, best_candidate, best_scores, history = attack_output
        non_nominated_front_fitness = None

    non_nominated_front_fitness = _sorted_front_by_score1(non_nominated_front_fitness)

    adv_chw = adv_chw.detach().cpu()
    weighted_fitness = best_scores.get("weighted_fitness")
    if weighted_fitness is None:
        weighted_fitness = args.w_margin * float(best_scores["margin_loss"]) + args.w_saliency * float(best_scores["saliency_loss"])

    clean_image_path = args.clean_image_output
    if clean_image_path is None:
        output_root, output_ext = os.path.splitext(args.output)
        output_ext = output_ext if output_ext else ".png"
        clean_image_path = f"{output_root}_clean{output_ext}"

    os.makedirs(os.path.dirname(clean_image_path) or ".", exist_ok=True)
    save_image(x_tensor[0].detach().cpu(), clean_image_path)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_image(adv_chw, args.output)

    with torch.no_grad():
        adv_pred = model(normalize(adv_chw.unsqueeze(0).to(device))).argmax(dim=1).item()

    clean_map_path = args.clean_map_output
    adv_map_path = args.adv_map_output
    if clean_map_path is None or adv_map_path is None:
        output_root, output_ext = os.path.splitext(args.output)
        output_ext = output_ext if output_ext else ".png"
        if clean_map_path is None:
            clean_map_path = f"{output_root}_clean_map{output_ext}"
        if adv_map_path is None:
            adv_map_path = f"{output_root}_adv_map{output_ext}"

    os.makedirs(os.path.dirname(clean_map_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(adv_map_path) or ".", exist_ok=True)

    output_root, output_ext = os.path.splitext(args.output)
    output_ext = output_ext if output_ext else ".png"
    four_map_paths = {
        "clean_a": f"{output_root}_clean_map_class_a{output_ext}",
        "clean_b": f"{output_root}_clean_map_class_b{output_ext}",
        "adv_a": f"{output_root}_adv_map_class_a{output_ext}",
        "adv_b": f"{output_root}_adv_map_class_b{output_ext}",
    }
    for path in four_map_paths.values():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    output_root, _ = os.path.splitext(args.output)
    non_dominated_front_txt = f"{output_root}_non_dominated_front_scores.txt"
    save_non_dominated_front_txt(non_nominated_front_fitness, non_dominated_front_txt)

    clean_saliency_map = fitness.saliency_true[0]
    adv_saliency_map, _ = explain_fn(model, adv_chw.unsqueeze(0).to(device), normalize, y_true)
    _save_saliency_map(clean_saliency_map, clean_map_path)
    _save_saliency_map(adv_saliency_map[0], adv_map_path)
    class_a = int(y_true.item())
    class_b = int(adv_pred)
    _save_four_class_maps(
        model=model,
        explain_fn=explain_fn,
        normalize=normalize,
        clean_chw=x_tensor[0],
        adv_chw=adv_chw,
        class_a=class_a,
        class_b=class_b,
        device=device,
        output_paths=four_map_paths,
    )

    margin_chart_path = None
    saliency_chart_path = None
    should_save_history_chart = (
        args.save_history_chart
        or (args.history_chart_output is not None)
        or (args.margin_chart_output is not None)
        or (args.saliency_chart_output is not None)
    )
    if should_save_history_chart:
        output_root, output_ext = os.path.splitext(args.output)
        output_ext = output_ext if output_ext else ".png"

        base_root = output_root
        if args.history_chart_output is not None:
            base_root, base_ext = os.path.splitext(args.history_chart_output)
            if base_ext:
                output_ext = base_ext

        margin_chart_path = args.margin_chart_output or f"{base_root}_margin{output_ext}"
        saliency_chart_path = args.saliency_chart_output or f"{base_root}_saliency{output_ext}"

        os.makedirs(os.path.dirname(margin_chart_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(saliency_chart_path) or ".", exist_ok=True)
        save_attack_two_score_charts(
            history,
            margin_output_path=margin_chart_path,
            saliency_output_path=saliency_chart_path,
        )

    print("=== Attack summary ===")
    print(f"model: {args.model}")
    print(f"image: {args.image}")
    print(f"true_label: {y_true}")
    print(f"clean_pred: {pred}")
    print(f"adv_pred: {adv_pred}")
    print(f"l0_distance: {int(best_candidate.l0_distance(adv_chw.to(device)))}")
    print(f"margin_loss: {float(best_scores['margin_loss'])}")
    print(f"saliency_loss: {float(best_scores['saliency_loss'])}")
    print(f"weighted_fitness: {float(weighted_fitness)}")
    print(f"first_success_iteration: {best_scores.get('first_success_iteration')}")
    print(f"algorithm: {args.algorithm}")
    print(f"fitness_function: {args.fitness_function}")
    print(f"operator_strategy: {args.operator_strategy}")
    print(f"saliency_temperature: {args.saliency_temperature}")
    print(f"seed: {args.seed}")
    print(f"saved_clean_image: {clean_image_path}")
    print(f"saved_adv: {args.output}")
    print(f"saved_clean_map: {clean_map_path}")
    print(f"saved_adv_map: {adv_map_path}")
    print(f"saved_clean_map_class_a: {four_map_paths['clean_a']}")
    print(f"saved_clean_map_class_b: {four_map_paths['clean_b']}")
    print(f"saved_adv_map_class_a: {four_map_paths['adv_a']}")
    print(f"saved_adv_map_class_b: {four_map_paths['adv_b']}")
    if non_nominated_front_fitness is not None:
        print(f"saved_non_dominated_front_scores: {non_dominated_front_txt}")
    if margin_chart_path is not None and saliency_chart_path is not None:
        print(f"saved_margin_chart: {margin_chart_path}")
        print(f"saved_saliency_chart: {saliency_chart_path}")
    else:
        print("saved_history_chart: disabled (use --save-history-chart)")

    return {
        "model": args.model,
        "image": args.image,
        "true_label": y_true.detach().cpu().tolist(),
        "clean_pred": pred.detach().cpu().tolist(),
        "adv_pred": adv_pred,
        "l0_distance": int(best_candidate.l0_distance(adv_chw.to(device))),
        "margin_loss": float(best_scores["margin_loss"]),
        "saliency_loss": float(best_scores["saliency_loss"]),
        "weighted_fitness": float(weighted_fitness),
        "first_success_iteration": best_scores.get("first_success_iteration"),
        "algorithm": args.algorithm,
        "fitness_function": args.fitness_function,
        "operator_strategy": args.operator_strategy,
        "saliency_temperature": args.saliency_temperature,
        "seed": args.seed,
        "saved_clean_image": clean_image_path,
        "saved_adv": args.output,
        "saved_clean_map": clean_map_path,
        "saved_adv_map": adv_map_path,
        "saved_clean_map_class_a": four_map_paths["clean_a"],
        "saved_clean_map_class_b": four_map_paths["clean_b"],
        "saved_adv_map_class_a": four_map_paths["adv_a"],
        "saved_adv_map_class_b": four_map_paths["adv_b"],
        "saved_non_dominated_front_scores": non_dominated_front_txt if non_nominated_front_fitness is not None else None,
        "saved_margin_chart": margin_chart_path,
        "saved_saliency_chart": saliency_chart_path,
    }


def main():
    args = parse_args()
    run_attack(args)


if __name__ == "__main__":
    main()
