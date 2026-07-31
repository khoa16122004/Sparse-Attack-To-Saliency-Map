from __future__ import annotations


import torchvision.models as tv_models
from torchvision.models import get_model_weights
import torchvision.transforms as T
import numpy as np
import torch
from explain_method import (
    simple_gradient_map,
    integrated_gradients,
    input_gradient_map,
    grad_cam,
    raw_attention,
    attention_grad,
)
from torch import nn
import matplotlib.pyplot as plt
import os

def _to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


_DATASET_NUM_CLASSES = {
    "imagenet": 1000,
    "imagenet1k": 1000,
    "cifar10": 10,
    "cifar100": 100,
    "mnist": 10,
    "fashionmnist": 10,
    "svhn": 10,
    "caltech101": 101,
    "caltech256": 256,
}



def build_blur_substrate(gkern_or_kernel_size=11, kernel_size=11, kernel_sigma=5):
    if callable(gkern_or_kernel_size):
        gkern_fn = gkern_or_kernel_size
    else:
        from RISE.evaluation import gkern as gkern_fn

        kernel_sigma = kernel_size
        kernel_size = gkern_or_kernel_size

    kernel = gkern_fn(kernel_size, kernel_sigma)

    def blur_fn(x):
        kernel_on_device = kernel.to(device=x.device, dtype=x.dtype)
        return nn.functional.conv2d(x, kernel_on_device, padding=kernel_size // 2)

    return blur_fn

def split_transform_from_weights(weights):

    resize = weights.transforms().resize_size
    crop = weights.transforms().crop_size
    mean = weights.transforms().mean
    std = weights.transforms().std

    spatial = T.Compose([
        T.Resize(resize),
        T.CenterCrop(crop),
        T.ToTensor()
    ])

    normalize = T.Normalize(mean=mean, std=std)

    return spatial, normalize

def get_torchvision_model(
    model_name,
    pretrained=True,
    num_classes=None,
):
    model_fn = getattr(tv_models, model_name)

    if pretrained:
        weights_enum = get_model_weights(model_name).DEFAULT
        model = model_fn(weights=weights_enum)

        spatial, normalize = split_transform_from_weights(weights_enum)

        return model, spatial, normalize

    kwargs = {}
    if num_classes is not None:
        kwargs["num_classes"] = num_classes

    model = model_fn(weights=None, **kwargs)

    return model, None, None


def get_explainable_method(method_name):
    if method_name == "simple_gradient":
        explain_method = simple_gradient_map
    elif method_name == "integrated_gradients":
        explain_method = integrated_gradients
    elif method_name == "input_gradient":
        explain_method = input_gradient_map
    elif method_name == "grad_cam":
        explain_method = grad_cam
    elif method_name == "raw_attention":
        explain_method = raw_attention
    elif method_name == "attention_grad":
        explain_method = attention_grad
    
    else:
        raise ValueError(f"Unknown explainable method: {method_name}")

    return explain_method



class TorchvisionModelWrapper:
    def __init__(self, model, normalize, device):
        self.model = model
        self.normalize = normalize
        self.device = device

    def predict(self, x):
        x = x.to(self.device)
        x = self.normalize(x)
        with torch.no_gra():
            logits = self.model(x)
        return logits.detach().cpu()
    
    def __call__(self, x):
        return self.predict(x)


def get_intersection(clean_map, adv_map):
    clean_map = np.asarray(clean_map, dtype=np.float32)
    adv_map = np.asarray(adv_map, dtype=np.float32)
    inter = np.minimum(clean_map, adv_map).sum()
    union = np.maximum(clean_map, adv_map).sum() + 1e-12
    return float(inter / union)


def save_attack_history_chart(history, output_path, title="Attack score history"):
    if not history:
        raise ValueError("history is empty, cannot create chart")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save history chart") from exc

    iterations = list(range(len(history)))
    margin_losses = [_to_float(item["margin_loss"]) for item in history]
    saliency_losses = [_to_float(item["saliency_loss"]) for item in history]
    weighted_scores = [_to_float(item["weighted_fitness"]) for item in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, margin_losses, label="margin_loss", linewidth=2)
    ax.plot(iterations, saliency_losses, label="saliency_loss", linewidth=2)
    ax.plot(iterations, weighted_scores, label="weighted_fitness", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_single_score_chart(iterations, values, label, output_path, title):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save history chart") from exc

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, values, label=label, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_attack_two_score_charts(
    history,
    margin_output_path,
    saliency_output_path,
    margin_title="Margin loss history",
    saliency_title="Saliency loss history",
):
    if not history:
        raise ValueError("history is empty, cannot create chart")

    iterations = list(range(len(history)))
    margin_losses = [_to_float(item["margin_loss"]) for item in history]
    saliency_losses = [_to_float(item["saliency_loss"]) for item in history]

    _save_single_score_chart(
        iterations=iterations,
        values=margin_losses,
        label="margin_loss",
        output_path=margin_output_path,
        title=margin_title,
    )
    _save_single_score_chart(
        iterations=iterations,
        values=saliency_losses,
        label="saliency_loss",
        output_path=saliency_output_path,
        title=saliency_title,
    )


def normalize_causal_curve(scores, mode="del", eps=1e-8):
    """
    Normalize deletion/insertion curve so that only the shape is compared.

    Parameters
    ----------
    scores : array-like
        Probability curve.
    mode : {"del", "ins"}
    eps : float
        Prevent division by zero.

    Returns
    -------
    np.ndarray
        Normalized curve.
    """
    scores = np.asarray(scores, dtype=np.float32)

    if mode == "del":
        # start -> 1, end -> 0
        denom = max(scores[0] - scores[-1], eps)
        norm_scores = (scores - scores[-1]) / denom

    elif mode == "ins":
        # start -> 0, end -> 1
        denom = max(scores[-1] - scores[0], eps)
        norm_scores = (scores - scores[0]) / denom

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return np.clip(norm_scores, 0.0, 1.0)

def save_causal_metric_summary(image_tensor, final_tensor, scores, output_path, mode, class_name, preprocess):
    if mode == "del":
        title = "Deletion game"
        ylabel = "Pixels deleted"
    elif mode == "ins":
        title = "Insertion game"
        ylabel = "Pixels inserted"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    n_steps = len(scores) - 1
    preview_image = denormalize_image_tensor(final_tensor, preprocess)
    
    # new normalize
    scores = normalize_causal_curve(scores, mode)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title(f"{ylabel} 100.0%, P={scores[-1]:.4f}")
    plt.axis("off")
    plt.imshow(preview_image)

    plt.subplot(122)
    plt.plot(np.arange(n_steps + 1) / n_steps, scores)
    plt.xlim(-0.1, 1.1)
    plt.ylim(0, 1.05)
    plt.fill_between(np.arange(n_steps + 1) / n_steps, 0, scores, alpha=0.4)
    plt.title(title)
    plt.xlabel(ylabel)
    plt.ylabel(class_name)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def denormalize_image_tensor(image_tensor, preprocess):
    mean, std = preprocess.mean, preprocess.std
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    image = image_tensor.detach().cpu().numpy().transpose((1, 2, 0))
    image = std * image + mean
    return np.clip(image, 0, 1)
