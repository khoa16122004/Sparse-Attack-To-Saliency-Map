import argparse
import json
import os
import random
import sys

import torch
from PIL import Image
from torchvision.utils import save_image


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(ROOT_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from explain_method_backprop import get_explainable_method_backprop
from spgd import SaliencySparsePGD
from util import get_torchvision_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run saliency-based sparse gradient attack on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="sbgd_adv.png", help="Path to save adversarial image")
    parser.add_argument("--model", type=str, default="resnet50", help="Torchvision model name")
    parser.add_argument("--label", type=int, default=None, help="True class. If omitted, use clean prediction")
    parser.add_argument(
        "--explain-method",
        type=str,
        default="input_gradient",
        choices=["simple_gradient", "integrated_gradients", "input_gradient", "raw_attention", "attention_grad"],
        help="Backprop-capable saliency method",
    )

    parser.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    parser.add_argument("--k", type=int, default=50, help="Top-k sparse coordinates")
    parser.add_argument("--sparsity-ratio", type=float, default=None, help="Sparse ratio p in (0, 1]")
    parser.add_argument("--tau", type=float, default=0.5, help="Threshold tau for binary sparse mask")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=1.0 / 255.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--attack-mode", type=str, default="pixel", choices=["pixel", "feature"])
    parser.add_argument("--w-margin", type=float, default=1.0)
    parser.add_argument("--w-saliency", type=float, default=1.0)
    parser.add_argument("--disable-softplus-surrogate", action="store_true", help="Use original model for saliency instead of Softplus surrogate")
    parser.add_argument("--softplus-beta", type=float, default=10.0, help="Softplus beta for surrogate model")
    parser.add_argument("--disable-fixed-mask-location", action="store_true", help="Allow mask location to vary during optimization")
    parser.add_argument("--zero-grad-patience", type=int, default=3, help="Iterations of near-zero grad before jitter")
    parser.add_argument("--zero-grad-jitter", type=float, default=1e-2, help="Jitter scale when escaping zero-gradient plateaus")
    parser.add_argument("--debug-grad", action="store_true", help="Print per-iteration gradient diagnostics")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--clean-map-output", type=str, default=None)
    parser.add_argument("--adv-map-output", type=str, default=None)
    parser.add_argument("--summary-output", type=str, default=None)
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

    from PIL import Image as PILImage

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    PILImage.fromarray(rgb, mode="RGB").save(output_path)


def _soft_iou(s1, s2, eps=1e-12):
    s1 = s1.flatten(start_dim=1)
    s2 = s2.flatten(start_dim=1)
    inter = torch.minimum(s1, s2).sum(dim=1)
    union = torch.maximum(s1, s2).sum(dim=1)
    return inter / (union + eps)


def _margin_loss(logits, y_true):
    true_logits = logits.gather(1, y_true.view(-1, 1)).squeeze(1)
    others = logits.clone()
    others.scatter_(1, y_true.view(-1, 1), float("-inf"))
    other_logits = others.max(dim=1).values
    return -(true_logits - other_logits)


def run(args):
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

    explain_fn = get_explainable_method_backprop(args.explain_method)

    image = Image.open(args.image).convert("RGB")
    x = spatial(image).to(device).unsqueeze(0)

    with torch.no_grad():
        clean_logits = model(normalize(x))
        clean_pred = clean_logits.argmax(dim=1)

    y_true = clean_pred if args.label is None else torch.tensor([args.label], device=device, dtype=torch.long)

    attacker = SaliencySparsePGD(
        model=model,
        normalize=normalize,
        explain_method=explain_fn,
        epsilon=args.epsilon,
        k=args.k,
        sparsity_ratio=args.sparsity_ratio,
        tau=args.tau,
        t=args.iterations,
        alpha=args.alpha,
        beta=args.beta,
        attack_mode=args.attack_mode,
        w_margin=args.w_margin,
        w_saliency=args.w_saliency,
        fixed_mask_location=not args.disable_fixed_mask_location,
        use_softplus_surrogate=not args.disable_softplus_surrogate,
        softplus_beta=args.softplus_beta,
        zero_grad_patience=args.zero_grad_patience,
        zero_grad_jitter=args.zero_grad_jitter,
        debug_grad=args.debug_grad,
    )

    x_adv, history = attacker.attack(x, y_true, return_history=True)

    save_image(x_adv[0].detach().cpu(), args.output)

    clean_saliency, _ = explain_fn(model, x, normalize, target_class=y_true)
    adv_saliency, adv_logits = explain_fn(model, x_adv, normalize, target_class=y_true)
    adv_pred = adv_logits.argmax(dim=1)

    clean_map_output = args.clean_map_output
    adv_map_output = args.adv_map_output
    if clean_map_output is None or adv_map_output is None:
        root, ext = os.path.splitext(args.output)
        ext = ext if ext else ".png"
        clean_map_output = clean_map_output or f"{root}_clean_map{ext}"
        adv_map_output = adv_map_output or f"{root}_adv_map{ext}"

    _save_saliency_map(clean_saliency[0], clean_map_output)
    _save_saliency_map(adv_saliency[0], adv_map_output)

    summary_output = args.summary_output
    if summary_output is None:
        root, _ = os.path.splitext(args.output)
        summary_output = f"{root}_summary.json"

    margin = _margin_loss(adv_logits, y_true).mean().item()
    soft_iou = _soft_iou(clean_saliency, adv_saliency).mean().item()

    payload = {
        "image": args.image,
        "model": args.model,
        "label_used": int(y_true[0].item()),
        "clean_pred": int(clean_pred[0].item()),
        "adv_pred": int(adv_pred[0].item()),
        "margin_loss": float(margin),
        "saliency_loss_softiou": float(soft_iou),
        "objective": float(args.w_margin * margin + args.w_saliency * soft_iou),
        "iters": int(args.iterations),
        "k": int(args.k),
        "epsilon": float(args.epsilon),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "history": history,
        "output_adv": args.output,
        "output_clean_map": clean_map_output,
        "output_adv_map": adv_map_output,
    }

    os.makedirs(os.path.dirname(summary_output) or ".", exist_ok=True)
    with open(summary_output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Clean pred:", int(clean_pred[0].item()))
    print("Adv pred  :", int(adv_pred[0].item()))
    print("Margin    :", float(margin))
    print("SoftIoU   :", float(soft_iou))
    print("Saved adv :", args.output)
    print("Saved sum :", summary_output)


if __name__ == "__main__":
    run(parse_args())
