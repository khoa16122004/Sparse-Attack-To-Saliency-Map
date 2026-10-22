# import torch
# import torch.nn.functional as F
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image


# def _prepare_target_class(output, target_class):
#     # Ensure one class index per sample in the current batch.
#     if target_class is None:
#         return output.argmax(dim=1)

#     if not isinstance(target_class, torch.Tensor):
#         target_class = torch.tensor(target_class, device=output.device)

#     target_class = target_class.to(device=output.device, dtype=torch.long).view(-1)
#     if target_class.numel() == 1 and output.size(0) > 1:
#         target_class = target_class.expand(output.size(0))

#     if target_class.numel() != output.size(0):
#         raise ValueError(
#             f"target_class has {target_class.numel()} elements, expected batch size {output.size(0)}"
#         )

#     return target_class



# def simple_gradient_map(model, input_tensor, normalize, target_class=None):
#     x = input_tensor.clone().detach()
#     x.requires_grad_(True) # b x 3 x w x h
#     model.zero_grad()

#     output = model(normalize(x))
#     output_logits = output.detach()
#     target_class = _prepare_target_class(output, target_class)

#     # gather scores for each sample
#     score = output.gather(1, target_class.view(-1,1)).sum()

#     score.backward()

#     grad = x.grad

#     # sum RGB
#     saliency = grad.abs().sum(dim=1)

#     H, W = saliency.shape[-2:]

#     # normalize per image
#     saliency = (H*W) * saliency / (saliency.view(saliency.size(0), -1).sum(dim=1).view(-1,1,1) + 1e-8)

#     return saliency.detach(), output_logits


# def input_gradient_map(model, input_tensor, normalize, target_class=None):
#     x = input_tensor.clone().detach()
#     x.requires_grad_(True) # b x 3 x w x h
#     model.zero_grad()

#     output = model(normalize(x))
#     output_logits = output.detach()
#     target_class = _prepare_target_class(output, target_class)

#     # gather scores for each sample
#     score = output.gather(1, target_class.view(-1,1)).sum()

#     score.backward()

#     grad = x.grad

#     # sum RGB
#     # x \odot grad
#     saliency = (x * grad).abs().sum(dim=1)    

#     H, W = saliency.shape[-2:]

#     # normalize per image
#     saliency = (H*W) * saliency / (saliency.view(saliency.size(0), -1).sum(dim=1).view(-1,1,1) + 1e-8)

#     return saliency.detach(), output_logits


# def integrated_gradients(model, input_tensor, normalize, target_class=None, steps=5, baseline=None):

#     model.eval()

#     x = input_tensor.clone().detach()
#     B = x.size(0)

#     if baseline is None:
#         baseline = torch.zeros_like(x)

#     with torch.no_grad():
#         output_ref = model(normalize(x))

#     if target_class is None:
#         target_class = output_ref.argmax(dim=1)

#     target_class = _prepare_target_class(output_ref, target_class)

#     grads = torch.zeros_like(x)

#     for i in range(1, steps+1):

#         alpha = float(i)/steps
#         inp = baseline + alpha * (x - baseline)

#         inp.requires_grad_(True)

#         model.zero_grad()

#         output = model(normalize(inp))
#         output_logits = output.detach()

#         score = output.gather(1, target_class.view(-1,1)).sum()

#         score.backward()

#         grads += inp.grad.detach()

#     avg_grad = grads / steps

#     ig = (x - baseline) * avg_grad

#     saliency = ig.abs().sum(dim=1)

#     H, W = saliency.shape[-2:]

#     saliency = (H*W) * saliency / (saliency.view(B,-1).sum(dim=1).view(-1,1,1) + 1e-8)

#     return saliency.detach(), output_logits

# def _infer_gradcam_model_name(model):
#     if hasattr(model, "layer4"):
#         return "resnet"

#     class_name = model.__class__.__name__.lower()
#     if class_name.startswith("vgg"):
#         return "vgg"
#     if class_name.startswith("densenet"):
#         return "densenet"

#     if class_name == "visiontransformer" or hasattr(model, "encoder"):
#         patch_size = getattr(model, "patch_size", None)
#         if patch_size == 16:
#             return "vit_b_16"
#         if patch_size == 32:
#             return "vit_b_32"

#     raise ValueError(
#         f"Could not infer Grad-CAM model name for class {model.__class__.__name__}. "
#         "Pass a supported torchvision model (resnet/vgg/densenet/vit_b_16/vit_b_32)."
#     )


# def grad_cam(model, input_tensor, normalize, target_class=None, model_name=None):
#     if model_name is None:
#         model_name = _infer_gradcam_model_name(model)

#     grayscale_cam, output_logits = get_gradcam_map(
#         model=model,
#         model_name=model_name,
#         input_tensor=input_tensor,
#         normalize=normalize,
#         target_class=target_class,
#     )

#     saliency = torch.as_tensor(
#         grayscale_cam,
#         device=input_tensor.device,
#         dtype=input_tensor.dtype,
#     )

#     if saliency.dim() == 2:
#         saliency = saliency.unsqueeze(0)

#     H, W = saliency.shape[-2:]
#     saliency = (H * W) * saliency / (
#         saliency.view(saliency.size(0), -1).sum(dim=1).view(-1, 1, 1) + 1e-8
#     )

#     return saliency.detach(), output_logits


# # For ViT
# # Raw attention
# # Rollout

# def _is_vit_model(model, model_name=None):
#     if model_name is not None:
#         return str(model_name).lower().startswith("vit")

#     class_name = model.__class__.__name__.lower()
#     return class_name == "visiontransformer" or hasattr(model, "encoder")


# def _is_torchvision_vit_model(model):
#     class_name = model.__class__.__name__.lower()
#     return (
#         class_name == "visiontransformer"
#         and hasattr(model, "_process_input")
#         and hasattr(model, "class_token")
#         and hasattr(model, "encoder")
#         and hasattr(model, "heads")
#     )


# def _forward_torchvision_vit_with_attentions(model, x):
#     # Re-implement torchvision ViT forward so we can request attention weights per block.
#     tokens = model._process_input(x)
#     batch_size = tokens.shape[0]
#     class_token = model.class_token.expand(batch_size, -1, -1)
#     tokens = torch.cat([class_token, tokens], dim=1)

#     encoder = model.encoder
#     tokens = tokens + encoder.pos_embedding
#     tokens = encoder.dropout(tokens)

#     attentions = []
#     for block in encoder.layers:
#         residual = tokens
#         x_norm = block.ln_1(tokens)
#         attn_out, attn_weights = block.self_attention(
#             x_norm,
#             x_norm,
#             x_norm,
#             need_weights=True,
#             average_attn_weights=False,
#         )
#         attn_out = block.dropout(attn_out)
#         tokens = residual + attn_out

#         y = block.ln_2(tokens)
#         y = block.mlp(y)
#         tokens = tokens + y

#         if attn_weights is None:
#             raise ValueError("Could not extract attention weights from torchvision ViT block.")
#         attentions.append(attn_weights)

#     tokens = encoder.ln(tokens)
#     logits = model.heads(tokens[:, 0])
#     return logits, tuple(attentions)


# def _forward_with_attentions(model, x):
#     try:
#         outputs = model(x, output_attentions=True)
#     except TypeError as exc:
#         if _is_torchvision_vit_model(model):
#             return _forward_torchvision_vit_with_attentions(model, x)
#         raise ValueError(
#             "This ViT model does not expose attentions via output_attentions=True. "
#             "Use a ViT implementation that returns attentions (e.g. HuggingFace ViTModel/ViTForImageClassification)."
#         ) from exc

#     if hasattr(outputs, "logits") and hasattr(outputs, "attentions"):
#         logits = outputs.logits
#         attentions = outputs.attentions
#         return logits, attentions

#     if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
#         logits = outputs[0]
#         attentions = outputs[-1]
#         return logits, attentions

#     raise ValueError("Could not parse logits/attentions from model outputs.")


# def _tokens_to_map(tokens, batch_size, out_hw):
#     grid = int(tokens.shape[-1] ** 0.5)
#     if grid * grid != tokens.shape[-1]:
#         raise ValueError("Number of patch tokens is not a perfect square.")

#     saliency = tokens.reshape(batch_size, 1, grid, grid)
#     saliency = F.interpolate(
#         saliency,
#         size=out_hw,
#         mode="bilinear",
#         align_corners=False,
#     ).squeeze(1)

#     saliency = saliency / (saliency.mean(dim=(1, 2), keepdim=True) + 1e-8)
#     return saliency


# def raw_attention(model, input_tensor, normalize, target_class=None, model_name=None):
#     if not _is_vit_model(model, model_name):
#         raise ValueError("raw_attention only supports ViT models.")

#     model.zero_grad()
#     x = input_tensor.clone().detach()

#     logits, attentions = _forward_with_attentions(model, normalize(x))
#     output_logits = logits.detach()

#     last_attn = attentions[-1]  # B x num_heads x num_tokens x num_tokens
#     attn_map = last_attn.mean(dim=1)  # B x num_tokens x num_tokens
#     cls_to_patches = attn_map[:, 0, 1:]  # B x (num_tokens - 1)

#     saliency = _tokens_to_map(
#         cls_to_patches,
#         batch_size=x.shape[0],
#         out_hw=x.shape[-2:],
#     )

#     return saliency.detach(), output_logits


# def attention_grad(model, input_tensor, normalize, target_class=None, model_name=None):
#     if not _is_vit_model(model, model_name):
#         raise ValueError("attention_grad only supports ViT models.")

#     model.zero_grad()
#     x = input_tensor.clone().detach().requires_grad_(True)

#     logits, attentions = _forward_with_attentions(model, normalize(x))
#     output_logits = logits
#     target_class = _prepare_target_class(logits, target_class)
#     score = logits.gather(1, target_class.view(-1, 1)).sum()

#     cams = []
#     for attn in attentions:
#         grad = torch.autograd.grad(score, attn, retain_graph=True, allow_unused=True)[0]
#         if grad is None:
#             continue

#         # Gradient-weighted attention, averaged across heads.
#         cam = (attn * grad).clamp(min=0).mean(dim=1)  # B x num_tokens x num_tokens
#         cams.append(cam)

#     if not cams:
#         raise ValueError(
#             "Attention tensors are not connected to logits for gradient computation. "
#             "Use a ViT implementation with differentiable returned attentions."
#         )

#     rollout = torch.eye(cams[0].shape[-1], device=x.device).unsqueeze(0)
#     rollout = rollout.repeat(x.shape[0], 1, 1)

#     for cam in cams:
#         cam = cam + torch.eye(cam.shape[-1], device=x.device)
#         cam = cam / (cam.sum(dim=-1, keepdim=True) + 1e-8)
#         rollout = cam @ rollout

#     cls_to_patches = rollout[:, 0, 1:]
#     saliency = _tokens_to_map(
#         cls_to_patches,
#         batch_size=x.shape[0],
#         out_hw=x.shape[-2:],
#     )

#     return saliency.detach(), output_logits



# def _vit_reshape_transform_vit_b_32(tensor, weight=7, height=7):
#     tensor = tensor[:, 1:, :]
#     tensor = tensor.reshape(tensor.size(0), weight, height, tensor.size(2))
#     return tensor.permute(0, 3, 1, 2)
    
# def _vit_reshape_transform_vit_b_16(tensor, weight=14, height=14):
#     tensor = tensor[:, 1:, :]
#     tensor = tensor.reshape(tensor.size(0), weight, height, tensor.size(2))
#     return tensor.permute(0, 3, 1, 2)


# def get_gradcam_target_layer(model, model_name):
#     model_name = model_name.lower()

#     if model_name.startswith("resnet"):
#         return [model.layer4[-1]], None

#     if model_name.startswith("vgg"):
#         return [model.features[-1]], None

#     if model_name.startswith("vit"):
#         if model_name == "vit_b_32":
#             return [model.encoder.layers[-1].ln_1], _vit_reshape_transform_vit_b_32
#         elif model_name == "vit_b_16":
#             return [model.encoder.layers[-1].ln_1], _vit_reshape_transform_vit_b_16
    
#     if model_name.startswith("densenet"):
#         return [model.features[-1]], None

#     raise ValueError(f"Grad-CAM target layer is not configured for model {model_name}")


# def get_gradcam_map(model, model_name, input_tensor, normalize, target_class=None):
#     model.eval()

#     with torch.no_grad():
#         logits = model(normalize(input_tensor))


#     target_layers, reshape_transform = get_gradcam_target_layer(model, model_name)
    
#     if target_class is None:
#         targets = None
#     else:
#         if isinstance(target_class, torch.Tensor):
#             target_values = target_class.detach().view(-1).tolist()
#         elif isinstance(target_class, (list, tuple)):
#             target_values = list(target_class)
#         else:
#             target_values = [target_class] * int(input_tensor.size(0))

#         if len(target_values) == 1 and input_tensor.size(0) > 1:
#             target_values = target_values * int(input_tensor.size(0))

#         targets = [ClassifierOutputTarget(int(t)) for t in target_values]



#     cam_kwargs = {
#         "model": model,
#         "target_layers": target_layers,
#     }
#     if reshape_transform is not None:
#         cam_kwargs["reshape_transform"] = reshape_transform

#     with GradCAM(**cam_kwargs) as cam:
#         grayscale_cam = cam(input_tensor=normalize(input_tensor), targets=targets)

#     output_logits = cam.outputs.detach()

#     return grayscale_cam, output_logits




# ==================== new explain method ==============
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

try:
    from pytorch_grad_cam import (
        AblationCAM,
        EigenCAM,
        EigenGradCAM,
        FullGrad,
        GradCAM,
        GradCAMPlusPlus,
        HiResCAM,
        LayerCAM,
        ScoreCAM,
        XGradCAM,
    )
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ModuleNotFoundError:
    # Fallback to bundled source tree so Grad-CAM can run without pip install.
    _THIS_DIR = Path(__file__).resolve().parent
    _LOCAL_CANDIDATES = [
        _THIS_DIR / "pytorch-grad-cam",
        _THIS_DIR.parent / "pytorch-grad-cam",
    ]
    for _candidate in _LOCAL_CANDIDATES:
        if _candidate.exists() and str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))

    from pytorch_grad_cam import (
        AblationCAM,
        EigenCAM,
        EigenGradCAM,
        FullGrad,
        GradCAM,
        GradCAMPlusPlus,
        HiResCAM,
        LayerCAM,
        ScoreCAM,
        XGradCAM,
    )
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image


_CAM_METHODS = {
    "grad_cam": GradCAM,
    "grad_cam_plusplus": GradCAMPlusPlus,
    "xgrad_cam": XGradCAM,
    "layer_cam": LayerCAM,
    "hires_cam": HiResCAM,
    "score_cam": ScoreCAM,
    "ablation_cam": AblationCAM,
    "eigen_cam": EigenCAM,
    "eigen_grad_cam": EigenGradCAM,
    "fullgrad": FullGrad,
}


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

    if class_name == "visiontransformer" or hasattr(model, "encoder") or hasattr(model, "blocks"):
        patch_size = getattr(model, "patch_size", None)
        if isinstance(patch_size, (tuple, list)) and len(patch_size) > 0:
            patch_size = patch_size[0]
        if patch_size == 16:
            return "vit_b_16"
        if patch_size == 32:
            return "vit_b_32"
        return "vit"

    raise ValueError(
        f"Could not infer Grad-CAM model name for class {model.__class__.__name__}. "
        "Pass a supported torchvision model (resnet/vgg/densenet/vit_b_16/vit_b_32)."
    )


def grad_cam(model, input_tensor, normalize, target_class=None, model_name=None):
    return cam_method_map(
        model=model,
        input_tensor=input_tensor,
        normalize=normalize,
        target_class=target_class,
        model_name=model_name,
        method_name="grad_cam",
    )


def cam_method_map(model, input_tensor, normalize, target_class=None, model_name=None, method_name="grad_cam"):
    if model_name is None:
        model_name = _infer_gradcam_model_name(model)

    grayscale_cam, output_logits = get_gradcam_map(
        model=model,
        model_name=model_name,
        input_tensor=input_tensor,
        normalize=normalize,
        target_class=target_class,
        method_name=method_name,
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


def grad_cam_plusplus(model, input_tensor, normalize, target_class=None, model_name=None):
    return cam_method_map(model, input_tensor, normalize, target_class, model_name, method_name="grad_cam_plusplus")


def xgrad_cam(model, input_tensor, normalize, target_class=None, model_name=None):
    return cam_method_map(model, input_tensor, normalize, target_class, model_name, method_name="xgrad_cam")


def layer_cam(model, input_tensor, normalize, target_class=None, model_name=None):
    return cam_method_map(model, input_tensor, normalize, target_class, model_name, method_name="layer_cam")


def hires_cam(model, input_tensor, normalize, target_class=None, model_name=None):
    return cam_method_map(model, input_tensor, normalize, target_class, model_name, method_name="hires_cam")


def score_cam(model, input_tensor, normalize, target_class=None, model_name=None):
    return cam_method_map(model, input_tensor, normalize, target_class, model_name, method_name="score_cam")


def ablation_cam(model, input_tensor, normalize, target_class=None, model_name=None):
    return cam_method_map(model, input_tensor, normalize, target_class, model_name, method_name="ablation_cam")


def eigen_cam(model, input_tensor, normalize, target_class=None, model_name=None):
    return cam_method_map(model, input_tensor, normalize, target_class, model_name, method_name="eigen_cam")


def eigen_grad_cam(model, input_tensor, normalize, target_class=None, model_name=None):
    return cam_method_map(model, input_tensor, normalize, target_class, model_name, method_name="eigen_grad_cam")


def fullgrad(model, input_tensor, normalize, target_class=None, model_name=None):
    return cam_method_map(model, input_tensor, normalize, target_class, model_name, method_name="fullgrad")


# For ViT
# Raw attention
# Rollout

def _is_vit_model(model, model_name=None):
    if model_name is not None:
        return str(model_name).lower().startswith("vit")

    class_name = model.__class__.__name__.lower()
    return class_name == "visiontransformer" or hasattr(model, "encoder") or hasattr(model, "blocks")


def _is_swin_model(model, model_name=None):
    if model_name is not None:
        return str(model_name).lower().startswith("swin")

    class_name = model.__class__.__name__.lower()
    return class_name == "swintransformer" or (
        hasattr(model, "features") and hasattr(model, "head") and hasattr(model, "avgpool")
    )


def _supports_attention_saliency(model, model_name=None):
    return _is_vit_model(model, model_name) or _is_swin_model(model, model_name)


def _is_torchvision_vit_model(model):
    class_name = model.__class__.__name__.lower()
    return (
        class_name == "visiontransformer"
        and hasattr(model, "_process_input")
        and hasattr(model, "class_token")
        and hasattr(model, "encoder")
        and hasattr(model, "heads")
    )


def _torchvision_self_attention(block, tokens):
    """Run torchvision self-attention while retaining differentiable weights."""
    attention = block.self_attention
    batch_size, num_tokens, embed_dim = tokens.shape
    num_heads = attention.num_heads
    head_dim = embed_dim // num_heads

    qkv = F.linear(tokens, attention.in_proj_weight, attention.in_proj_bias)
    query, key, value = qkv.chunk(3, dim=-1)
    query = query.reshape(batch_size, num_tokens, num_heads, head_dim).transpose(1, 2)
    key = key.reshape(batch_size, num_tokens, num_heads, head_dim).transpose(1, 2)
    value = value.reshape(batch_size, num_tokens, num_heads, head_dim).transpose(1, 2)

    attentions = (query * (head_dim ** -0.5)) @ key.transpose(-2, -1)
    attentions = attentions.softmax(dim=-1)
    if attention.dropout > 0 and attention.training:
        attentions = F.dropout(attentions, p=attention.dropout)

    attn_out = attentions @ value
    attn_out = attn_out.transpose(1, 2).reshape(batch_size, num_tokens, embed_dim)
    attn_out = attention.out_proj(attn_out)
    return attn_out, attentions


def _forward_torchvision_vit_with_attentions(model, x):
    # Re-implement torchvision ViT forward so attention weights stay in the graph.
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
        attn_out, attn_weights = _torchvision_self_attention(block, x_norm)
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


def _forward_torchvision_swin_with_feature_map(model, x):
    # Torchvision Swin pipeline: features -> norm -> permute -> avgpool -> head.
    feat = model.features(x)
    if hasattr(model, "norm"):
        feat = model.norm(feat)

    if feat.dim() != 4:
        raise ValueError("Expected 4D features from Swin model.")

    # Keep both layouts: BHWC for head path and BCHW for saliency map.
    if feat.shape[-1] > feat.shape[1]:
        feat_bhwc = feat
        feat_bchw = feat.permute(0, 3, 1, 2)
    else:
        feat_bchw = feat
        feat_bhwc = feat.permute(0, 2, 3, 1)

    # Use BCHW features directly for pooling so the returned feature map is
    # guaranteed to be on the logits computation path.
    pooled_input = feat_bchw

    pooled = model.avgpool(pooled_input)
    logits = model.head(torch.flatten(pooled, 1))
    return logits, feat_bchw


def raw_attention(model, input_tensor, normalize, target_class=None, model_name=None):
    if not _supports_attention_saliency(model, model_name):
        raise ValueError("raw_attention only supports ViT/Swin models.")

    model.zero_grad()
    x = input_tensor.clone().detach()

    if _is_swin_model(model, model_name):
        logits, feat_map = _forward_torchvision_swin_with_feature_map(model, normalize(x))
        output_logits = logits.detach()

        # Swin has no class token attention matrix; use final feature energy as token saliency.
        saliency = feat_map.pow(2).mean(dim=1)
        saliency = F.interpolate(
            saliency.unsqueeze(1),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        saliency = saliency / (saliency.mean(dim=(1, 2), keepdim=True) + 1e-8)
        return saliency.detach(), output_logits

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
    if not _supports_attention_saliency(model, model_name):
        raise ValueError("attention_grad only supports ViT/Swin models.")

    model.zero_grad()
    x = input_tensor.clone().detach().requires_grad_(True)

    if _is_swin_model(model, model_name):
        logits, feat_map = _forward_torchvision_swin_with_feature_map(model, normalize(x))
        output_logits = logits
        target_class = _prepare_target_class(logits, target_class)
        score = logits.gather(1, target_class.view(-1, 1)).sum()

        grad = torch.autograd.grad(score, feat_map, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            raise ValueError("Could not compute gradients from logits to Swin feature map.")

        # Rollout-style fallback for Swin: gradient-weighted final token map.
        saliency = (feat_map * grad).sum(dim=1).clamp(min=0)
        saliency = F.interpolate(
            saliency.unsqueeze(1),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        saliency = saliency / (saliency.mean(dim=(1, 2), keepdim=True) + 1e-8)
        return saliency.detach(), output_logits

    logits, attentions = _forward_with_attentions(model, normalize(x))
    output_logits = logits
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



def _vit_reshape_transform(tensor, height=None, width=None):
    # Remove class token and reshape patch tokens to a spatial map.
    result = tensor[:, 1:, :]

    if height is None or width is None:
        num_patches = result.size(1)
        side = int(num_patches ** 0.5)
        if side * side != num_patches:
            raise ValueError(
                f"Cannot infer square feature map from {num_patches} patch tokens. "
                "Pass a ViT input size that yields a square token grid."
            )
        height = side
        width = side

    result = result.reshape(result.size(0), height, width, result.size(2))

    # Bring channels to first dimension (B, C, H, W), like CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def _vit_reshape_transform_vit_b_32(tensor, width=7, height=7):
    return _vit_reshape_transform(tensor, height=height, width=width)


def _vit_reshape_transform_vit_b_16(tensor, width=14, height=14):
    return _vit_reshape_transform(tensor, height=height, width=width)


def _normalize_patch_size(patch_size):
    if isinstance(patch_size, (tuple, list)) and len(patch_size) > 0:
        return int(patch_size[0])
    if patch_size is None:
        return None
    return int(patch_size)


def _resolve_vit_gradcam_target_layer(model):
    # torchvision VisionTransformer (e.g. vit_b_16 / vit_b_32)
    if _is_torchvision_vit_model(model):
        return model.encoder.layers[-1].ln_1

    # timm / DeiT / many ViT implementations
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        last_block = model.blocks[-1]
        if hasattr(last_block, "norm1"):
            return last_block.norm1

    # torchvision ViT fallback
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers") and len(model.encoder.layers) > 0:
        last_block = model.encoder.layers[-1]
        if hasattr(last_block, "ln_1"):
            return last_block.ln_1
        if hasattr(last_block, "norm1"):
            return last_block.norm1

    raise ValueError(
        "Could not resolve ViT Grad-CAM target layer. Expected model.blocks[-1].norm1 "
        "or model.encoder.layers[-1].ln_1."
    )


def _resolve_vit_reshape_transform(model, model_name):
    if _is_torchvision_vit_model(model):
        patch_size = _normalize_patch_size(getattr(model, "patch_size", None))
        if patch_size == 32:
            return _vit_reshape_transform_vit_b_32
        return _vit_reshape_transform_vit_b_16

    if model_name == "vit_b_32":
        return _vit_reshape_transform_vit_b_32
    if model_name == "vit_b_16":
        return _vit_reshape_transform_vit_b_16

    patch_size = _normalize_patch_size(getattr(model, "patch_size", None))
    if patch_size == 32:
        return _vit_reshape_transform_vit_b_32
    if patch_size == 16:
        return _vit_reshape_transform_vit_b_16
    return _vit_reshape_transform


def get_gradcam_target_layer(model, model_name):
    model_name = model_name.lower()

    if model_name.startswith("resnet"):
        return [model.layer4[-1]], None

    if model_name.startswith("vgg"):
        return [model.features[-1]], None

    if model_name.startswith("vit"):
        return [_resolve_vit_gradcam_target_layer(model)], _resolve_vit_reshape_transform(model, model_name)
    
    if model_name.startswith("densenet"):
        return [model.features[-1]], None

    raise ValueError(f"Grad-CAM target layer is not configured for model {model_name}")


def get_gradcam_map(model, model_name, input_tensor, normalize, target_class=None, method_name="grad_cam"):
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

    cam_cls = _CAM_METHODS.get(method_name)
    if cam_cls is None:
        supported = ", ".join(sorted(_CAM_METHODS.keys()))
        raise ValueError(f"Unknown CAM method '{method_name}'. Supported methods: {supported}")

    with cam_cls(**cam_kwargs) as cam:
        grayscale_cam = cam(input_tensor=normalize(input_tensor), targets=targets)

    output_logits = cam.outputs.detach()

    return grayscale_cam, output_logits


