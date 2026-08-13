import argparse
import sys
from pathlib import Path

from PIL import Image
import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))

from explain_method import _forward_with_attentions, attention_grad
from explain_method_backprop import attention_grad as attention_grad_backprop
from util import get_explainable_method, get_torchvision_model


def _save_saliency_map(saliency, output_path):
    saliency = saliency.detach().float().cpu()
    saliency = saliency - saliency.min()
    saliency = saliency / (saliency.max() + 1e-8)

    red = (saliency * 3.0).clamp(0.0, 1.0)
    green = (saliency * 3.0 - 1.0).clamp(0.0, 1.0)
    blue = (saliency * 3.0 - 2.0).clamp(0.0, 1.0)
    rgb = torch.stack((red, green, blue), dim=-1).mul(255).byte().numpy()
    Image.fromarray(rgb, mode="RGB").save(output_path)


def run_image(model_name, image_path, explain_method, output_map, device, checkpoint_dir=None):
    model, spatial, normalize = get_torchvision_model(
        model_name,
        pretrained=True,
        checkpoint_dir=checkpoint_dir,
    )
    model = model.to(device).eval()

    image = Image.open(image_path).convert("RGB")
    input_tensor = spatial(image).unsqueeze(0).to(device)
    explain = get_explainable_method(explain_method)
    target_class = None
    saliency, logits = explain(model, input_tensor, normalize, target_class)

    output_path = Path(output_map)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_saliency_map(saliency[0], output_path)

    probabilities = logits.softmax(dim=1)[0]
    scores, classes = probabilities.topk(5)
    print(f"model: {model_name}")
    print(f"image: {image_path}")
    print(f"explain_method: {explain_method}")
    print(f"output_map: {output_path}")
    print("top5_predictions:")
    for rank, (class_id, score) in enumerate(zip(classes.tolist(), scores.tolist()), start=1):
        print(f"  {rank}. class={class_id} probability={score:.6f}")


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run a torchvision explanation on one image")
    parser.add_argument("--model", required=True, help="Torchvision model name, for example vit_b_32")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--explain-method",
        required=True,
        choices=["attention_grad", "raw_attention"],
        help="Explanation method",
    )
    parser.add_argument("--output-map", default="vit_explanation.png", help="Output saliency map path")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory containing <model>.pth")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_cli_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")
    run_image(
        model_name=args.model,
        image_path=args.image,
        explain_method=args.explain_method,
        output_map=args.output_map,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )


@pytest.fixture
def vit_b32():
    model = torchvision.models.vit_b_32(weights=None).eval()
    return model, torch.randn(1, 3, 224, 224)


def test_torchvision_vit_attention_rollout_is_differentiable(vit_b32):
    model, input_tensor = vit_b32

    logits, attentions = _forward_with_attentions(model, input_tensor)
    reference_logits = model(input_tensor)
    score = logits[:, logits.argmax(dim=1)].sum()
    gradient = torch.autograd.grad(score, attentions[-1])[0]

    assert torch.allclose(logits, reference_logits, atol=1e-5, rtol=1e-4)
    assert attentions[-1].shape == (1, 12, 50, 50)
    assert gradient is not None


@pytest.mark.parametrize("explain", [attention_grad, attention_grad_backprop])
def test_torchvision_vit_attention_grad_returns_image_map(vit_b32, explain):
    model, input_tensor = vit_b32

    saliency, logits = explain(model, input_tensor, lambda value: value)

    assert saliency.shape == (1, 224, 224)
    assert logits.shape == (1, 1000)
    assert torch.isfinite(saliency).all()


if __name__ == "__main__":
    main()