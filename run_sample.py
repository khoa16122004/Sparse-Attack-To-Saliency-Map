import argparse
import os
import sys

import torch
from PIL import Image
from torchvision.utils import save_image


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(ROOT_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from LossFunctions import MarginSalinecy_Fitness
from util import get_explainable_method, get_torchvision_model, save_attack_history_chart
from weightedSUM_GA import Weighted_Sum_GA


def parse_args():
    parser = argparse.ArgumentParser(description="Run sparse attack with weighted-sum genetic algorithm")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="adv.png", help="Path to save adversarial image")
    parser.add_argument("--clean-image-output", type=str, default=None, help="Path to save resized clean image")
    parser.add_argument("--clean-map-output", type=str, default=None, help="Path to save clean/original saliency map")
    parser.add_argument("--adv-map-output", type=str, default=None, help="Path to save adversarial saliency map")
    parser.add_argument("--save-history-chart", action="store_true", help="Save chart of margin/saliency/weighted scores")
    parser.add_argument("--history-chart-output", type=str, default=None, help="Path to save history chart")
    parser.add_argument("--model", type=str, default="resnet50", help="Torchvision model name")
    parser.add_argument("--label", type=int, default=None, help="True label index. If omitted, uses model prediction")
    parser.add_argument(
        "--explain-method",
        type=str,
        default="simple_gradient",
        choices=["simple_gradient", "integrated_gradients"],
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
        "--operator-strategy",
        type=str,
        default="uniform",
        choices=["uniform", "saliency_guided"],
        help="Genetic operator strategy for init/crossover/mutation",
    )
    parser.add_argument(
        "--saliency-temperature",
        type=float,
        default=1.0,
        help="Temperature for saliency-guided pixel sampling (lower is sharper)",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

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


def main():
    args = parse_args()
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

    fitness = MarginSalinecy_Fitness(
        model=model,
        x_tensor=x_tensor,
        normalize=normalize,
        y_true=y_true,
        explain_method=explain_fn,
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

    attacker = Weighted_Sum_GA(ga_params)
    adv_chw, best_candidate, best_scores, history = attacker.attack()
    adv_chw = adv_chw.detach().cpu()

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

    clean_saliency_map = fitness.saliency_true[0]
    adv_saliency_map, _ = explain_fn(model, adv_chw.unsqueeze(0).to(device), normalize, y_true)
    _save_saliency_map(clean_saliency_map, clean_map_path)
    _save_saliency_map(adv_saliency_map[0], adv_map_path)

    history_chart_path = None
    should_save_history_chart = args.save_history_chart or (args.history_chart_output is not None)
    if should_save_history_chart:
        history_chart_path = args.history_chart_output
        if history_chart_path is None:
            output_root, output_ext = os.path.splitext(args.output)
            output_ext = output_ext if output_ext else ".png"
            history_chart_path = f"{output_root}_scores{output_ext}"

        os.makedirs(os.path.dirname(history_chart_path) or ".", exist_ok=True)
        save_attack_history_chart(history, history_chart_path)

    print("=== Attack summary ===")
    print(f"model: {args.model}")
    print(f"image: {args.image}")
    print(f"true_label: {y_true}")
    print(f"clean_pred: {pred}")
    print(f"adv_pred: {adv_pred}")
    print(f"l0_distance: {int(best_candidate.l0_distance(adv_chw.to(device)))}")
    print(f"margin_loss: {float(best_scores['margin_loss'])}")
    print(f"saliency_loss: {float(best_scores['saliency_loss'])}")
    print(f"weighted_fitness: {float(best_scores['weighted_fitness'])}")
    print(f"operator_strategy: {args.operator_strategy}")
    print(f"saliency_temperature: {args.saliency_temperature}")
    print(f"saved_clean_image: {clean_image_path}")
    print(f"saved_adv: {args.output}")
    print(f"saved_clean_map: {clean_map_path}")
    print(f"saved_adv_map: {adv_map_path}")
    if history_chart_path is not None:
        print(f"saved_history_chart: {history_chart_path}")
    else:
        print("saved_history_chart: disabled (use --save-history-chart)")


if __name__ == "__main__":
    main()
