import torch
import torch.nn.functional as F
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
    saliency = (H*W) * saliency / (saliency.view(saliency.size(0), -1).sum(dim=1).view(-1,1,1) + 1e-8)

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
    saliency = (H*W) * saliency / (saliency.view(saliency.size(0), -1).sum(dim=1).view(-1,1,1) + 1e-8)

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

    saliency = (H*W) * saliency / (saliency.view(B,-1).sum(dim=1).view(-1,1,1) + 1e-8)

    return saliency.detach(), output_logits

def _infer_gradcam_model_name(model):
    if hasattr(model, "layer4"):
        return "resnet"

    class_name = model.__class__.__name__.lower()
    if class_name.startswith("vgg"):
        return "vgg"
    if class_name.startswith("densenet"):
        return "densenet"

    if class_name == "visiontransformer" or hasattr(model, "encoder"):
        patch_size = getattr(model, "patch_size", None)
        if patch_size == 16:
            return "vit_b_16"
        if patch_size == 32:
            return "vit_b_32"

    raise ValueError(
        f"Could not infer Grad-CAM model name for class {model.__class__.__name__}. "
        "Pass a supported torchvision model (resnet/vgg/densenet/vit_b_16/vit_b_32)."
    )


def grad_cam(model, input_tensor, normalize, target_class=None, model_name=None):
    if model_name is None:
        model_name = _infer_gradcam_model_name(model)

    grayscale_cam, output_logits = get_gradcam_map(
        model=model,
        model_name=model_name,
        input_tensor=input_tensor,
        normalize=normalize,
        target_class=target_class,
    )

    saliency = torch.as_tensor(
        grayscale_cam,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )

    if saliency.dim() == 2:
        saliency = saliency.unsqueeze(0)

    H, W = saliency.shape[-2:]
    saliency = (H * W) * saliency / (
        saliency.view(saliency.size(0), -1).sum(dim=1).view(-1, 1, 1) + 1e-8
    )

    return saliency.detach(), output_logits


# For ViT
# Raw attention
# Rollout

def _is_vit_model(model, model_name=None):
    if model_name is not None:
        return str(model_name).lower().startswith("vit")

    class_name = model.__class__.__name__.lower()
    return class_name == "visiontransformer" or hasattr(model, "encoder")


def _is_torchvision_vit_model(model):
    class_name = model.__class__.__name__.lower()
    return (
        class_name == "visiontransformer"
        and hasattr(model, "_process_input")
        and hasattr(model, "class_token")
        and hasattr(model, "encoder")
        and hasattr(model, "heads")
    )


def _forward_torchvision_vit_with_attentions(model, x):
    # Re-implement torchvision ViT forward so we can request attention weights per block.
    tokens = model._process_input(x)
    batch_size = tokens.shape[0]
    class_token = model.class_token.expand(batch_size, -1, -1)
    tokens = torch.cat([class_token, tokens], dim=1)

    encoder = model.encoder
    tokens = tokens + encoder.pos_embedding
    tokens = encoder.dropout(tokens)

    attentions = []
    for block in encoder.layers:
        residual = tokens
        x_norm = block.ln_1(tokens)
        attn_out, attn_weights = block.self_attention(
            x_norm,
            x_norm,
            x_norm,
            need_weights=True,
            average_attn_weights=False,
        )
        attn_out = block.dropout(attn_out)
        tokens = residual + attn_out

        y = block.ln_2(tokens)
        y = block.mlp(y)
        tokens = tokens + y

        if attn_weights is None:
            raise ValueError("Could not extract attention weights from torchvision ViT block.")
        attentions.append(attn_weights)

    tokens = encoder.ln(tokens)
    logits = model.heads(tokens[:, 0])
    return logits, tuple(attentions)


def _forward_with_attentions(model, x):
    try:
        outputs = model(x, output_attentions=True)
    except TypeError as exc:
        if _is_torchvision_vit_model(model):
            return _forward_torchvision_vit_with_attentions(model, x)
        raise ValueError(
            "This ViT model does not expose attentions via output_attentions=True. "
            "Use a ViT implementation that returns attentions (e.g. HuggingFace ViTModel/ViTForImageClassification)."
        ) from exc

    if hasattr(outputs, "logits") and hasattr(outputs, "attentions"):
        logits = outputs.logits
        attentions = outputs.attentions
        return logits, attentions

    if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        logits = outputs[0]
        attentions = outputs[-1]
        return logits, attentions

    raise ValueError("Could not parse logits/attentions from model outputs.")


def _tokens_to_map(tokens, batch_size, out_hw):
    grid = int(tokens.shape[-1] ** 0.5)
    if grid * grid != tokens.shape[-1]:
        raise ValueError("Number of patch tokens is not a perfect square.")

    saliency = tokens.reshape(batch_size, 1, grid, grid)
    saliency = F.interpolate(
        saliency,
        size=out_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    saliency = saliency / (saliency.mean(dim=(1, 2), keepdim=True) + 1e-8)
    return saliency


def raw_attention(model, input_tensor, normalize, target_class=None, model_name=None):
    if not _is_vit_model(model, model_name):
        raise ValueError("raw_attention only supports ViT models.")

    model.zero_grad()
    x = input_tensor.clone().detach()

    logits, attentions = _forward_with_attentions(model, normalize(x))
    output_logits = logits.detach()

    last_attn = attentions[-1]  # B x num_heads x num_tokens x num_tokens
    attn_map = last_attn.mean(dim=1)  # B x num_tokens x num_tokens
    cls_to_patches = attn_map[:, 0, 1:]  # B x (num_tokens - 1)

    saliency = _tokens_to_map(
        cls_to_patches,
        batch_size=x.shape[0],
        out_hw=x.shape[-2:],
    )

    return saliency.detach(), output_logits


def attention_grad(model, input_tensor, normalize, target_class=None, model_name=None):
    if not _is_vit_model(model, model_name):
        raise ValueError("attention_grad only supports ViT models.")

    model.zero_grad()
    x = input_tensor.clone().detach().requires_grad_(True)

    logits, attentions = _forward_with_attentions(model, normalize(x))
    output_logits = logits.detach()
    target_class = _prepare_target_class(logits, target_class)
    score = logits.gather(1, target_class.view(-1, 1)).sum()

    cams = []
    for attn in attentions:
        grad = torch.autograd.grad(score, attn, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            continue

        # Gradient-weighted attention, averaged across heads.
        cam = (attn * grad).clamp(min=0).mean(dim=1)  # B x num_tokens x num_tokens
        cams.append(cam)

    if not cams:
        raise ValueError(
            "Attention tensors are not connected to logits for gradient computation. "
            "Use a ViT implementation with differentiable returned attentions."
        )

    rollout = torch.eye(cams[0].shape[-1], device=x.device).unsqueeze(0)
    rollout = rollout.repeat(x.shape[0], 1, 1)

    for cam in cams:
        cam = cam + torch.eye(cam.shape[-1], device=x.device)
        cam = cam / (cam.sum(dim=-1, keepdim=True) + 1e-8)
        rollout = cam @ rollout

    cls_to_patches = rollout[:, 0, 1:]
    saliency = _tokens_to_map(
        cls_to_patches,
        batch_size=x.shape[0],
        out_hw=x.shape[-2:],
    )

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


