import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.utils import save_image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(ROOT_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from RISE.evaluation import CausalMetric, auc
from explain_method_backprop import get_explainable_method_backprop
from pgd_sparse import PGDSparseAttacker
from imagenet_metadata import IMAGENET_CLASSNAMES
from util import (
    build_blur_substrate,
    get_torchvision_model,
    save_attack_two_score_charts,
    save_causal_metric_summary,
)


class SoftmaxModel(nn.Module):
    def __init__(self, model, dim=1):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, inputs):
        return self.softmax(self.model(inputs))


def parse_args():
    parser = argparse.ArgumentParser(description="Run causal PGD_sparse attack on one sample")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="adv.png", help="Path to save adversarial image")
    parser.add_argument("--clean-image-output", type=str, default=None)
    parser.add_argument("--clean-map-output", type=str, default=None)
    parser.add_argument("--adv-map-output", type=str, default=None)
    parser.add_argument("--summary-output", type=str, default=None)
    parser.add_argument("--history-output", type=str, default=None)
    parser.add_argument("--save-history-chart", action="store_true")
    parser.add_argument("--history-chart-output", type=str, default=None)
    parser.add_argument("--margin-chart-output", type=str, default=None)
    parser.add_argument("--saliency-chart-output", type=str, default=None)

    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--label", type=int, default=None)
    parser.add_argument(
        "--explain-method",
        type=str,
        default="simple_gradient",
        choices=[
            "simple_gradient",
            "integrated_gradients",
            "input_gradient",
            "raw_attention",
            "attention_grad",
        ],
    )

    parser.add_argument("--iterations", type=int, default=80, help="Number of PGD_sparse iterations")
    parser.add_argument("--alpha", type=float, default=1.0, help="Step size for dense latent perturbation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold tau for sparsification")
    parser.add_argument("--lambda-margin", type=float, default=0.7, help="Weight lambda in -lambda*Lmargin + (1-lambda)*Lsal")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--autocast", action="store_true", help="Use autocast while computing forward/saliency")
    parser.add_argument("--autocast-dtype", type=str, default="float16", choices=["float16", "bfloat16"])

    parser.add_argument("--step", type=int, default=224)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--kernel-sigma", type=int, default=5)
    parser.add_argument("--save-process", action="store_true")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])

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


def _save_history_scores_txt(history, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in history:
            f.write(f"{float(item['margin_loss']):.12g} {float(item['saliency_loss']):.12g}\n")


def _compute_causal_metrics(model, normalize, clean_batch, adv_batch, clean_saliency_map, adv_saliency_map, output_root, args):
    metric_model = SoftmaxModel(model)

    blur_fn = build_blur_substrate(args.kernel_size, args.kernel_sigma)
    insertion = CausalMetric(metric_model, "ins", args.step, substrate_fn=blur_fn)
    deletion = CausalMetric(metric_model, "del", args.step, substrate_fn=lambda x: torch.zeros_like(x))

    clean_del_steps_dir = output_root / "clean_deletion_steps"
    clean_ins_steps_dir = output_root / "clean_insertion_steps"
    adv_del_steps_dir = output_root / "adv_deletion_steps"
    adv_ins_steps_dir = output_root / "adv_insertion_steps"

    if args.save_process:
        clean_del_steps_dir.mkdir(parents=True, exist_ok=True)
        clean_ins_steps_dir.mkdir(parents=True, exist_ok=True)
        adv_del_steps_dir.mkdir(parents=True, exist_ok=True)
        adv_ins_steps_dir.mkdir(parents=True, exist_ok=True)

    clean_scores_del = deletion.single_run(
        normalize(clean_batch).cpu().detach(),
        clean_saliency_map.cpu().detach().numpy(),
        verbose=args.verbose,
        save_to=str(clean_del_steps_dir) if args.save_process else None,
    )
    clean_scores_ins = insertion.single_run(
        normalize(clean_batch).cpu().detach(),
        clean_saliency_map.cpu().detach().numpy(),
        verbose=args.verbose,
        save_to=str(clean_ins_steps_dir) if args.save_process else None,
    )

    adv_scores_del = deletion.single_run(
        normalize(adv_batch).cpu().detach(),
        clean_saliency_map.cpu().detach().numpy(),
        verbose=args.verbose,
        save_to=str(adv_del_steps_dir) if args.save_process else None,
    )
    adv_scores_ins = insertion.single_run(
        normalize(adv_batch).cpu().detach(),
        adv_saliency_map[0].cpu().detach().numpy(),
        verbose=args.verbose,
        save_to=str(adv_ins_steps_dir) if args.save_process else None,
    )

    return {
        "clean_del_auc": float(auc(clean_scores_del)),
        "clean_ins_auc": float(auc(clean_scores_ins)),
        "adv_del_auc": float(auc(adv_scores_del)),
        "adv_ins_auc": float(auc(adv_scores_ins)),
        "clean_del_scores": [float(v) for v in clean_scores_del],
        "clean_ins_scores": [float(v) for v in clean_scores_ins],
        "adv_del_scores": [float(v) for v in adv_scores_del],
        "adv_ins_scores": [float(v) for v in adv_scores_ins],
    }


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

    explain_fn = get_explainable_method_backprop(args.explain_method)

    autocast_dtype = torch.float16 if args.autocast_dtype == "float16" else torch.bfloat16
    attacker = PGDSparseAttacker(
        model=model,
        normalize=normalize,
        explain_method=explain_fn,
        x_tensor=x_tensor,
        y_true=y_true,
        step_size=args.alpha,
        iterations=args.iterations,
        threshold=args.threshold,
        weight_margin=args.lambda_margin,
        use_autocast=args.autocast,
        autocast_dtype=autocast_dtype,
    )
    result = attacker.attack()

    output_path = Path(args.output)
    output_root = output_path.parent
    output_root.mkdir(parents=True, exist_ok=True)

    clean_image_path = Path(args.clean_image_output) if args.clean_image_output else output_root / f"{output_path.stem}_clean{output_path.suffix or '.png'}"
    clean_map_path = Path(args.clean_map_output) if args.clean_map_output else output_root / f"{output_path.stem}_clean_map{output_path.suffix or '.png'}"
    adv_map_path = Path(args.adv_map_output) if args.adv_map_output else output_root / f"{output_path.stem}_adv_map{output_path.suffix or '.png'}"

    history_txt_path = Path(args.history_output) if args.history_output else output_root / f"{output_path.stem}_history_scores.txt"
    history_json_path = output_root / f"{output_path.stem}_pgd_history.json"
    summary_path = Path(args.summary_output) if args.summary_output else output_root / "summary.json"

    four_map_paths = {
        "clean_a": str(output_root / f"{output_path.stem}_clean_map_class_a{output_path.suffix or '.png'}"),
        "clean_b": str(output_root / f"{output_path.stem}_clean_map_class_b{output_path.suffix or '.png'}"),
        "adv_a": str(output_root / f"{output_path.stem}_adv_map_class_a{output_path.suffix or '.png'}"),
        "adv_b": str(output_root / f"{output_path.stem}_adv_map_class_b{output_path.suffix or '.png'}"),
    }

    save_image(x_tensor[0].detach().cpu(), str(clean_image_path))
    save_image(result.adv_chw.detach().cpu(), str(output_path))

    with torch.no_grad():
        adv_pred = int(model(normalize(result.adv_chw.unsqueeze(0).to(device))).argmax(dim=1).item())

    clean_saliency_map, _ = explain_fn(model, x_tensor, normalize, y_true)
    adv_saliency_map, _ = explain_fn(model, result.adv_chw.unsqueeze(0).to(device), normalize, y_true)
    _save_saliency_map(clean_saliency_map[0], str(clean_map_path))
    _save_saliency_map(adv_saliency_map[0], str(adv_map_path))

    _save_four_class_maps(
        model=model,
        explain_fn=explain_fn,
        normalize=normalize,
        clean_chw=x_tensor[0],
        adv_chw=result.adv_chw,
        class_a=int(y_true.item()),
        class_b=adv_pred,
        device=device,
        output_paths=four_map_paths,
    )

    _save_history_scores_txt(result.history, str(history_txt_path))
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(result.history, f, indent=2, ensure_ascii=False)

    torch.save(result.dense_delta.cpu(), output_root / f"{output_path.stem}_dense_delta.pt")
    torch.save(result.delta_soft.cpu(), output_root / f"{output_path.stem}_delta_soft.pt")
    torch.save(result.delta_sparse.cpu(), output_root / f"{output_path.stem}_delta_sparse.pt")
    torch.save(result.sparse_mask.cpu(), output_root / f"{output_path.stem}_sparse_mask.pt")

    should_save_history_chart = (
        args.save_history_chart
        or (args.history_chart_output is not None)
        or (args.margin_chart_output is not None)
        or (args.saliency_chart_output is not None)
    )
    if should_save_history_chart:
        output_ext = output_path.suffix or ".png"
        base_root = output_root / output_path.stem
        if args.history_chart_output is not None:
            candidate = Path(args.history_chart_output)
            base_root = candidate.with_suffix("")
            if candidate.suffix:
                output_ext = candidate.suffix

        margin_chart_path = Path(args.margin_chart_output) if args.margin_chart_output else Path(f"{base_root}_margin{output_ext}")
        saliency_chart_path = Path(args.saliency_chart_output) if args.saliency_chart_output else Path(f"{base_root}_saliency{output_ext}")
        margin_chart_path.parent.mkdir(parents=True, exist_ok=True)
        saliency_chart_path.parent.mkdir(parents=True, exist_ok=True)

        save_attack_two_score_charts(
            result.history,
            margin_output_path=str(margin_chart_path),
            saliency_output_path=str(saliency_chart_path),
        )

    causal = _compute_causal_metrics(
        model=model,
        normalize=normalize,
        clean_batch=x_tensor,
        adv_batch=result.adv_chw.unsqueeze(0),
        clean_saliency_map=clean_saliency_map[0],
        adv_saliency_map=adv_saliency_map,
        output_root=output_root,
        args=args,
    )

    save_causal_metric_summary(
        image_tensor=normalize(x_tensor),
        final_tensor=torch.zeros_like(normalize(x_tensor)),
        scores=causal["clean_del_scores"],
        output_path=str(output_root / "clean_del_summary.png"),
        mode="del",
        class_name=IMAGENET_CLASSNAMES[int(pred.item())],
        preprocess=normalize,
    )
    save_causal_metric_summary(
        image_tensor=normalize(x_tensor),
        final_tensor=normalize(x_tensor),
        scores=causal["clean_ins_scores"],
        output_path=str(output_root / "clean_ins_summary.png"),
        mode="ins",
        class_name=IMAGENET_CLASSNAMES[int(pred.item())],
        preprocess=normalize,
    )
    save_causal_metric_summary(
        image_tensor=normalize(result.adv_chw.unsqueeze(0)),
        final_tensor=torch.zeros_like(normalize(result.adv_chw.unsqueeze(0))),
        scores=causal["adv_del_scores"],
        output_path=str(output_root / "adv_del_summary.png"),
        mode="del",
        class_name=IMAGENET_CLASSNAMES[int(pred.item())],
        preprocess=normalize,
    )
    save_causal_metric_summary(
        image_tensor=normalize(result.adv_chw.unsqueeze(0)),
        final_tensor=normalize(result.adv_chw.unsqueeze(0)),
        scores=causal["adv_ins_scores"],
        output_path=str(output_root / "adv_ins_summary.png"),
        mode="ins",
        class_name=IMAGENET_CLASSNAMES[int(pred.item())],
        preprocess=normalize,
    )

    summary = {
        "status": "ok",
        "model": args.model,
        "image": args.image,
        "true_label": int(y_true.item()),
        "clean_pred": int(pred.item()),
        "adv_pred": adv_pred,
        "algorithm": "pgd_sparse",
        "fitness_function": "pgd_sparse_joint",
        "explain_method": args.explain_method,
        "iterations": int(args.iterations),
        "alpha": float(args.alpha),
        "threshold": float(args.threshold),
        "lambda_margin": float(args.lambda_margin),
        "margin_loss": float(result.best_scores["margin_loss"]),
        "saliency_loss": float(result.best_scores["saliency_loss"]),
        "weighted_fitness": float(result.best_scores["weighted_fitness"]),
        "first_success_iteration": result.best_scores["first_success_iteration"],
        "l0_distance": int(result.best_scores["l0_distance"]),
        "sparse_ratio": float(result.best_scores["sparse_ratio"]),
        "saved_clean_image": str(clean_image_path),
        "saved_adv": str(output_path),
        "saved_clean_map": str(clean_map_path),
        "saved_adv_map": str(adv_map_path),
        "saved_history_txt": str(history_txt_path),
        "saved_history_json": str(history_json_path),
        "causual": {
            "del": causal["adv_del_scores"],
            "ins": causal["adv_ins_scores"],
            "clean_del": causal["clean_del_scores"],
            "clean_ins": causal["clean_ins_scores"],
            "auc": {
                "del": causal["adv_del_auc"],
                "ins": causal["adv_ins_auc"],
                "clean_del": causal["clean_del_auc"],
                "clean_ins": causal["clean_ins_auc"],
            },
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== PGD_sparse attack summary ===")
    print(f"model: {args.model}")
    print(f"image: {args.image}")
    print(f"clean_pred: {int(pred.item())}")
    print(f"adv_pred: {adv_pred}")
    print(f"margin_loss: {summary['margin_loss']:.6f}")
    print(f"saliency_loss: {summary['saliency_loss']:.6f}")
    print(f"weighted_fitness: {summary['weighted_fitness']:.6f}")
    print(f"first_success_iteration: {summary['first_success_iteration']}")
    print(f"l0_distance: {summary['l0_distance']}")
    print(f"sparse_ratio: {summary['sparse_ratio']:.6f}")
    print(f"summary: {summary_path}")


def main():
    args = parse_args()
    run_attack(args)


if __name__ == "__main__":
    main()
