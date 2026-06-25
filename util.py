from __future__ import annotations


import torchvision.models as tv_models
from torchvision.models import get_model_weights
import torchvision.transforms as T
import numpy as np
import torch
from explain_method import simple_gradient_map, integrated_gradients


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


def get_intersection(clean_map, adv_map):
    clean_map = np.asarray(clean_map, dtype=np.float32)
    adv_map = np.asarray(adv_map, dtype=np.float32)
    inter = np.minimum(clean_map, adv_map).sum()
    union = np.maximum(clean_map, adv_map).sum() + 1e-12
    return float(inter / union)


