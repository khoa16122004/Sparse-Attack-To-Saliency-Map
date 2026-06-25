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
from util import TorchvisionModelWrapper, get_explainable_method, get_torchvision_model
from weightedSUM_GA import Weighted_Sum_GA


def parse_args():
	parser = argparse.ArgumentParser(description="Run sparse attack with weighted-sum genetic algorithm")
	parser.add_argument("--image", type=str, required=True, help="Path to input image")
	parser.add_argument("--output", type=str, default="adv.png", help="Path to save adversarial image")
	parser.add_argument("--model", type=str, default="resnet50", help="Torchvision model name")
	parser.add_argument("--label", type=int, default=None, help="True label index. If omitted, uses model prediction")
	parser.add_argument(
		"--explain-method",
		type=str,
		default="simple_gradient",
		choices=["simple_gradient", "integrated_gradients"],
		help="Saliency explanation method",
	)

	parser.add_argument("--pop-size", type=int, default=40)
	parser.add_argument("--iterations", type=int, default=50)
	parser.add_argument("--eps", type=int, default=50, help="Number of perturbed pixels")
	parser.add_argument("--p-size", type=float, default=1.0, help="Perturbation step size")
	parser.add_argument("--pc", type=float, default=0.4, help="Crossover ratio")
	parser.add_argument("--pm", type=float, default=0.1, help="Mutation ratio")
	parser.add_argument("--zero-probability", type=float, default=0.3, help="Probability of zero channel perturbation")
	parser.add_argument("--w-margin", type=float, default=0.5)
	parser.add_argument("--w-saliency", type=float, default=0.5)
	parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

	return parser.parse_args()





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
    x_tensor = spatial(image).to(device).unsqueeze(0)  # Add batch dimension

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
        "device": args.device,
    }

    attacker = Weighted_Sum_GA(ga_params)
    adv_chw, best_candidate, best_scores, history = attacker.attack()
    adv_chw = adv_chw.detach().cpu()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_image(adv_chw, args.output)

    with torch.no_grad():
        adv_pred = model(adv_chw.unsqueeze(0).to(device)).argmax(dim=1).item()

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
    print(f"saved_adv: {args.output}")


if __name__ == "__main__":
	main()
