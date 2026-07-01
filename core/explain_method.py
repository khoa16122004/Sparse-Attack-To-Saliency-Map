import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def _prepare_target_class(output, target_class):
    # Ensure one class index per sample in the current batch.
    if target_class is None:
        return output.argmax(dim=1)

    if not isinstance(target_class, torch.Tensor):
        target_class = torch.tensor(target_class, device=output.device)

    target_class = target_class.to(device=output.device, dtype=torch.long).view(-1)
    if target_class.numel() == 1 and output.size(0) > 1:
        target_class = target_class.expand(output.size(0))

    if target_class.numel() != output.size(0):
        raise ValueError(
            f"target_class has {target_class.numel()} elements, expected batch size {output.size(0)}"
        )

    return target_class



def simple_gradient_map(model, input_tensor, normalize, target_class=None):
    x = input_tensor.clone().detach()
    x.requires_grad_(True) # b x 3 x w x h
    model.zero_grad()

    output = model(normalize(x))
    output_logits = output.detach()
    target_class = _prepare_target_class(output, target_class)

    # gather scores for each sample
    score = output.gather(1, target_class.view(-1,1)).sum()

    score.backward()

    grad = x.grad

    # sum RGB
    saliency = grad.abs().sum(dim=1)

    H, W = saliency.shape[-2:]

    # normalize per image
    # saliency = (H*W) * saliency / (saliency.view(saliency.size(0), -1).sum(dim=1).view(-1,1,1) + 1e-8)

    return saliency.detach(), output_logits


def input_gradient_map(model, input_tensor, normalize, target_class=None):
    x = input_tensor.clone().detach()
    x.requires_grad_(True) # b x 3 x w x h
    model.zero_grad()

    output = model(normalize(x))
    output_logits = output.detach()
    target_class = _prepare_target_class(output, target_class)

    # gather scores for each sample
    score = output.gather(1, target_class.view(-1,1)).sum()

    score.backward()

    grad = x.grad

    # sum RGB
    # x \odot grad
    saliency = (x * grad).abs().sum(dim=1)    

    H, W = saliency.shape[-2:]

    # normalize per image
    # saliency = (H*W) * saliency / (saliency.view(saliency.size(0), -1).sum(dim=1).view(-1,1,1) + 1e-8)

    return saliency.detach(), output_logits


def integrated_gradients(model, input_tensor, normalize, target_class=None, steps=5, baseline=None):

    model.eval()

    x = input_tensor.clone().detach()
    B = x.size(0)

    if baseline is None:
        baseline = torch.zeros_like(x)

    with torch.no_grad():
        output_ref = model(normalize(x))

    if target_class is None:
        target_class = output_ref.argmax(dim=1)

    target_class = _prepare_target_class(output_ref, target_class)

    grads = torch.zeros_like(x)

    for i in range(1, steps+1):

        alpha = float(i)/steps
        inp = baseline + alpha * (x - baseline)

        inp.requires_grad_(True)

        model.zero_grad()

        output = model(normalize(inp))
        output_logits = output.detach()

        score = output.gather(1, target_class.view(-1,1)).sum()

        score.backward()

        grads += inp.grad.detach()

    avg_grad = grads / steps

    ig = (x - baseline) * avg_grad

    saliency = ig.abs().sum(dim=1)

    H, W = saliency.shape[-2:]

    # saliency = (H*W) * saliency / (saliency.view(B,-1).sum(dim=1).view(-1,1,1) + 1e-8)

    return saliency.detach(), output_logits


def _vit_reshape_transform_vit_b_32(tensor, weight=7, height=7):
    tensor = tensor[:, 1:, :]
    tensor = tensor.reshape(tensor.size(0), weight, height, tensor.size(2))
    return tensor.permute(0, 3, 1, 2)
    
def _vit_reshape_transform_vit_b_16(tensor, weight=14, height=14):
    tensor = tensor[:, 1:, :]
    tensor = tensor.reshape(tensor.size(0), weight, height, tensor.size(2))
    return tensor.permute(0, 3, 1, 2)


def get_gradcam_target_layer(model, model_name):
    model_name = model_name.lower()

    if model_name.startswith("resnet"):
        return [model.layer4[-1]], None

    if model_name.startswith("vgg"):
        return [model.features[-1]], None

    if model_name.startswith("vit"):
        if model_name == "vit_b_32":
            return [model.encoder.layers[-1].ln_1], _vit_reshape_transform_vit_b_32
        elif model_name == "vit_b_16":
            return [model.encoder.layers[-1].ln_1], _vit_reshape_transform_vit_b_16
    
    if model_name.startswith("densenet"):
        return [model.features[-1]], None

    raise ValueError(f"Grad-CAM target layer is not configured for model {model_name}")


def get_gradcam_map(model, model_name, input_tensor, normalize, target_class=None):
    model.eval()

    with torch.no_grad():
        logits = model(normalize(input_tensor))


    target_layers, reshape_transform = get_gradcam_target_layer(model, model_name)
    
    if target_class is None:
        targets = None
    else:
        if isinstance(target_class, torch.Tensor):
            target_values = target_class.detach().view(-1).tolist()
        elif isinstance(target_class, (list, tuple)):
            target_values = list(target_class)
        else:
            target_values = [target_class] * int(input_tensor.size(0))

        if len(target_values) == 1 and input_tensor.size(0) > 1:
            target_values = target_values * int(input_tensor.size(0))

        targets = [ClassifierOutputTarget(int(t)) for t in target_values]



    cam_kwargs = {
        "model": model,
        "target_layers": target_layers,
    }
    if reshape_transform is not None:
        cam_kwargs["reshape_transform"] = reshape_transform

    with GradCAM(**cam_kwargs) as cam:
        grayscale_cam = cam(input_tensor=normalize(input_tensor), targets=targets)

    output_logits = cam.outputs.detach()

    return grayscale_cam, output_logits


