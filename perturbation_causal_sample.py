"""Treat the sparse adversarial perturbation as a saliency map and evaluate it with
insertion/deletion (RISE CausalMetric). Also evaluates the regular explanation
saliency maps (clean/adv) and stores every AUC in a single JSON file."""

import argparse
import json
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

from causual_sample import SoftmaxModel, _save_saliency_map, create_attacker, create_fitness
from util import get_explainable_method, get_torchvision_model, build_blur_substrate, save_causal_metric_summary
from RISE.evaluation import CausalMetric, auc
from imagenet_metadata import IMAGENET_CLASSNAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse attack + insertion/deletion on the perturbation map")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="perturbation_causal_output", help="Folder to save all results")
    parser.add_argument("--model", type=str, default="resnet50", help="Torchvision model name")
    parser.add_argument("--label", type=int, default=None, help="True label index. If omitted, uses model prediction")
    parser.add_argument(
        "--explain-method",
        type=str,
        default="simple_gradient",
        choices=["simple_gradient", "integrated_gradients", "input_gradient", "grad_cam", "raw_attention", "attention_grad"],
    )
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--eps", type=int, default=50, help="Number of perturbed pixels")
    parser.add_argument("--p-size", type=float, default=1.0, help="Perturbation step size")
    parser.add_argument("--pc", type=float, default=0.9)
    parser.add_argument("--pm", type=float, default=0.1)
    parser.add_argument("--zero-probability", type=float, default=0.3)
    parser.add_argument("--w-margin", type=float, default=0.5)
    parser.add_argument("--w-saliency", type=float, default=0.5)
    parser.add_argument(
        "--fitness-function",
        type=str,
        default="margin_saliency",
        choices=[
            "margin_saliency",
            "negative_cross_entropy_saliency",
            "cross_entropy_saliency",
            "reverse_margin_saliency",
            "reverse_negative_cross_entropy_saliency",
            "margin_loss_mse_saliency",
            "MarginLossMESE",
        ],
    )
    parser.add_argument("--algorithm", type=str, default="weighted_sum_ga", choices=["weighted_sum_ga", "nsgaii"])
    parser.add_argument("--operator-strategy", type=str, default="uniform", choices=["uniform", "saliency_guided"])
    parser.add_argument("--saliency-temperature", type=float, default=1.0)
    parser.add_argument(
        "--channel-reduce",
        type=str,
        default="sum",
        choices=["sum", "mean", "max", "l2"],
        help="How to reduce |x - x_adv| (H x W x 3) to a single channel",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--step", type=int, default=224, help="Pixels modified per causal-metric step")
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--kernel-sigma", type=int, default=5)
    parser.add_argument("--save-process", action="store_true", help="Save every deletion/insertion step image")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    return parser.parse_args()


def perturbation_to_saliency(x_clean_chw, x_adv_chw, reduce="sum"):
    """|x - x_adv| (3 x H x W) -> single-channel H x W map."""
    diff = (x_clean_chw.detach().cpu().float() - x_adv_chw.detach().cpu().float()).abs()
    if reduce == "sum":
        return diff.sum(dim=0)
    if reduce == "mean":
        return diff.mean(dim=0)
    if reduce == "max":
        return diff.amax(dim=0)
    if reduce == "l2":
        return diff.pow(2).sum(dim=0).sqrt()
    raise ValueError(f"Unsupported channel reduce: {reduce}")


def run_causal_pair(name, image_norm, saliency_2d, insertion, deletion, out_dir, normalize, args):
    """Run insertion + deletion of one (image, saliency) pair and save summary plots."""
    image_norm = image_norm.detach().cpu()
    saliency_np = saliency_2d.detach().cpu().float().numpy()

    with torch.no_grad():
        pred_cls = int(insertion.model(image_norm.to(next(insertion.model.parameters()).device)).argmax(dim=1).item())
    class_name = IMAGENET_CLASSNAMES[pred_cls]

    del_dir = os.path.join(out_dir, f"{name}_deletion_steps") if args.save_process else None
    ins_dir = os.path.join(out_dir, f"{name}_insertion_steps") if args.save_process else None
    for d in (del_dir, ins_dir):
        if d is not None:
            os.makedirs(d, exist_ok=True)

    del_curve = deletion.single_run(image_norm.clone(), saliency_np, verbose=args.verbose, save_to=del_dir)
    ins_curve = insertion.single_run(image_norm.clone(), saliency_np, verbose=args.verbose, save_to=ins_dir)

    save_causal_metric_summary(
        image_tensor=image_norm,
        final_tensor=torch.zeros_like(image_norm),
        scores=del_curve,
        output_path=os.path.join(out_dir, f"{name}_del_summary.png"),
        mode="del",
        class_name=class_name,
        preprocess=normalize,
    )
    save_causal_metric_summary(
        image_tensor=image_norm,
        final_tensor=image_norm,
        scores=ins_curve,
        output_path=os.path.join(out_dir, f"{name}_ins_summary.png"),
        mode="ins",
        class_name=class_name,
        preprocess=normalize,
    )

    del_auc = float(auc(del_curve))
    ins_auc = float(auc(ins_curve))
    print(f"[{name}] DEL AUC={del_auc:.6f} | INS AUC={ins_auc:.6f}")
    return {
        "evaluated_class": pred_cls,
        "evaluated_class_name": class_name,
        "del_auc": del_auc,
        "ins_auc": ins_auc,
        "del_curve": [float(v) for v in del_curve],
        "ins_curve": [float(v) for v in ins_curve],
    }


def run(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    model, spatial, normalize = get_torchvision_model(args.model, pretrained=True)
    model = model.to(device).eval()

    image = Image.open(args.image).convert("RGB")
    x_tensor = spatial(image).to(device).unsqueeze(0)

    with torch.no_grad():
        pred = model(normalize(x_tensor)).argmax(dim=1)
    y_true = pred if args.label is None else torch.tensor([args.label], device=device)

    explain_fn = get_explainable_method(args.explain_method)
    fitness = create_fitness(args.fitness_function, model, x_tensor, normalize, y_true, explain_fn)

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
    attack_output = create_attacker(ga_params, args.algorithm).attack()
    adv_chw, best_candidate = attack_output[0], attack_output[1]
    adv_chw = adv_chw.detach().cpu()
    clean_chw = x_tensor[0].detach().cpu()

    with torch.no_grad():
        adv_pred = int(model(normalize(adv_chw.unsqueeze(0).to(device))).argmax(dim=1).item())

    # (i) perturbation as saliency map
    perturbation_map = perturbation_to_saliency(clean_chw, adv_chw, args.channel_reduce)

    # (ii) explanation saliency maps
    clean_saliency_map = fitness.saliency_true[0].detach().cpu()
    adv_saliency_map, _ = explain_fn(model, adv_chw.unsqueeze(0).to(device), normalize, y_true)
    adv_saliency_map = adv_saliency_map[0].detach().cpu()

    save_image(clean_chw, os.path.join(out_dir, "clean.png"))
    save_image(adv_chw, os.path.join(out_dir, "adv.png"))
    _save_saliency_map(perturbation_map, os.path.join(out_dir, "perturbation_map.png"))
    _save_saliency_map(clean_saliency_map, os.path.join(out_dir, "clean_saliency_map.png"))
    _save_saliency_map(adv_saliency_map, os.path.join(out_dir, "adv_saliency_map.png"))
    torch.save(
        {
            "perturbation_map": perturbation_map,
            "clean_saliency_map": clean_saliency_map,
            "adv_saliency_map": adv_saliency_map,
        },
        os.path.join(out_dir, "maps.pt"),
    )

    metric_model = SoftmaxModel(model).eval()
    blur_fn = build_blur_substrate(args.kernel_size, args.kernel_sigma)
    insertion = CausalMetric(metric_model, "ins", args.step, substrate_fn=blur_fn)
    deletion = CausalMetric(metric_model, "del", args.step, substrate_fn=lambda x: torch.zeros_like(x))

    clean_norm = normalize(clean_chw.unsqueeze(0))
    adv_norm = normalize(adv_chw.unsqueeze(0))

    with torch.no_grad():
        results = {
            # (i) perturbation map used as saliency
            "perturbation_on_clean": run_causal_pair(
                "perturbation_on_clean", clean_norm, perturbation_map, insertion, deletion, out_dir, normalize, args
            ),
            "perturbation_on_adv": run_causal_pair(
                "perturbation_on_adv", adv_norm, perturbation_map, insertion, deletion, out_dir, normalize, args
            ),
            # (ii) explanation saliency maps
            "saliency_clean": run_causal_pair(
                "saliency_clean", clean_norm, clean_saliency_map, insertion, deletion, out_dir, normalize, args
            ),
            "saliency_adv": run_causal_pair(
                "saliency_adv", adv_norm, adv_saliency_map, insertion, deletion, out_dir, normalize, args
            ),
        }

    summary = {
        "image": args.image,
        "model": args.model,
        "explain_method": args.explain_method,
        "algorithm": args.algorithm,
        "fitness_function": args.fitness_function,
        "eps": args.eps,
        "seed": args.seed,
        "channel_reduce": args.channel_reduce,
        "step": args.step,
        "true_label": int(y_true.item()),
        "clean_pred": int(pred.item()),
        "adv_pred": adv_pred,
        "is_adversarial": adv_pred != int(y_true.item()),
        "l0_distance": int(best_candidate.l0_distance(adv_chw.to(device))),
        "auc": {k: {"ins_auc": v["ins_auc"], "del_auc": v["del_auc"]} for k, v in results.items()},
        "details": results,
    }

    json_path = os.path.join(out_dir, "causal_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved_json: {json_path}")
    return summary


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
