import torch
import torch.nn.functional as F

from explain_method import _forward_with_attentions, _is_vit_model, _tokens_to_map


def _ensure_grad_input(input_tensor):
    if input_tensor.requires_grad:
        return input_tensor
    return input_tensor.clone().detach().requires_grad_(True)


def _maybe_detach_outputs(saliency, logits, detach):
    if detach:
        return saliency.detach(), logits.detach()
    return saliency, logits


def _prepare_target_class(output, target_class):
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


def simple_gradient_map(model, input_tensor, normalize, target_class=None, create_graph=False, detach=True):
    x = _ensure_grad_input(input_tensor)
    model.zero_grad()

    output = model(normalize(x))
    output_logits = output
    target_class = _prepare_target_class(output, target_class)

    score = output.gather(1, target_class.view(-1, 1)).sum()
    grad = torch.autograd.grad(
        score,
        x,
        retain_graph=create_graph,
        create_graph=create_graph,
    )[0]

    saliency = grad.abs().sum(dim=1)
    h, w = saliency.shape[-2:]
    saliency = (h * w) * saliency / (saliency.view(saliency.size(0), -1).sum(dim=1).view(-1, 1, 1) + 1e-8)
    return _maybe_detach_outputs(saliency, output_logits, detach)


def input_gradient_map(model, input_tensor, normalize, target_class=None, create_graph=False, detach=True):
    x = _ensure_grad_input(input_tensor)
    model.zero_grad()

    output = model(normalize(x))
    output_logits = output
    target_class = _prepare_target_class(output, target_class)

    score = output.gather(1, target_class.view(-1, 1)).sum()
    grad = torch.autograd.grad(
        score,
        x,
        retain_graph=create_graph,
        create_graph=create_graph,
    )[0]

    saliency = (x * grad).abs().sum(dim=1)
    h, w = saliency.shape[-2:]
    saliency = (h * w) * saliency / (saliency.view(saliency.size(0), -1).sum(dim=1).view(-1, 1, 1) + 1e-8)
    return _maybe_detach_outputs(saliency, output_logits, detach)


def integrated_gradients(
    model,
    input_tensor,
    normalize,
    target_class=None,
    steps=5,
    baseline=None,
    create_graph=False,
    detach=True,
):
    model.eval()

    x = _ensure_grad_input(input_tensor)
    bsz = x.size(0)

    if baseline is None:
        baseline = torch.zeros_like(x)

    with torch.no_grad():
        output_ref = model(normalize(x))

    if target_class is None:
        target_class = output_ref.argmax(dim=1)
    target_class = _prepare_target_class(output_ref, target_class)

    grads = torch.zeros_like(x)
    output_logits = output_ref

    for i in range(1, steps + 1):
        alpha = float(i) / steps
        inp = baseline + alpha * (x - baseline)

        model.zero_grad()
        output = model(normalize(inp))
        output_logits = output

        score = output.gather(1, target_class.view(-1, 1)).sum()
        grad = torch.autograd.grad(
            score,
            inp,
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        grads = grads + grad

    avg_grad = grads / steps
    ig = (x - baseline) * avg_grad

    saliency = ig.abs().sum(dim=1)
    h, w = saliency.shape[-2:]
    saliency = (h * w) * saliency / (saliency.view(bsz, -1).sum(dim=1).view(-1, 1, 1) + 1e-8)
    return _maybe_detach_outputs(saliency, output_logits, detach)


def raw_attention(model, input_tensor, normalize, target_class=None, model_name=None, create_graph=False, detach=True):
    if not _is_vit_model(model, model_name):
        raise ValueError("raw_attention only supports ViT models.")

    model.zero_grad()
    x = _ensure_grad_input(input_tensor)

    logits, attentions = _forward_with_attentions(model, normalize(x))
    output_logits = logits

    last_attn = attentions[-1]
    attn_map = last_attn.mean(dim=1)
    cls_to_patches = attn_map[:, 0, 1:]

    saliency = _tokens_to_map(
        cls_to_patches,
        batch_size=x.shape[0],
        out_hw=x.shape[-2:],
    )

    return _maybe_detach_outputs(saliency, output_logits, detach)


def attention_grad(model, input_tensor, normalize, target_class=None, model_name=None, create_graph=False, detach=True):
    if not _is_vit_model(model, model_name):
        raise ValueError("attention_grad only supports ViT models.")

    model.zero_grad()
    x = _ensure_grad_input(input_tensor)

    logits, attentions = _forward_with_attentions(model, normalize(x))
    output_logits = logits
    target_class = _prepare_target_class(logits, target_class)
    score = logits.gather(1, target_class.view(-1, 1)).sum()

    cams = []
    for attn in attentions:
        grad = torch.autograd.grad(
            score,
            attn,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )[0]
        if grad is None:
            continue

        cam = (attn * grad).clamp(min=0).mean(dim=1)
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

    return _maybe_detach_outputs(saliency, output_logits, detach)


def get_explainable_method_backprop(method_name):
    if method_name == "simple_gradient":
        return simple_gradient_map
    if method_name == "integrated_gradients":
        return integrated_gradients
    if method_name == "input_gradient":
        return input_gradient_map
    if method_name == "raw_attention":
        return raw_attention
    if method_name == "attention_grad":
        return attention_grad
    raise ValueError(f"Unknown backprop explainable method: {method_name}")
