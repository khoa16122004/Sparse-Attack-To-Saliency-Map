from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class PGDSparseResult:
    adv_chw: torch.Tensor
    dense_delta: torch.Tensor
    delta_soft: torch.Tensor
    delta_sparse: torch.Tensor
    sparse_mask: torch.Tensor
    best_scores: Dict[str, float]
    history: List[Dict[str, float]]


class PGDSparseAttacker:
    """Projected-style sparse attack using a dense latent perturbation and thresholding."""

    def __init__(
        self,
        model: torch.nn.Module,
        normalize: Callable,
        explain_method: Callable,
        x_tensor: torch.Tensor,
        y_true: torch.Tensor,
        step_size: float,
        iterations: int,
        threshold: float,
        weight_margin: float,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        use_autocast: bool = True,
        autocast_dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.normalize = normalize
        self.explain_method = explain_method
        self.x_tensor = x_tensor
        self.y_true = y_true.long()
        self.step_size = float(step_size)
        self.iterations = int(iterations)
        self.threshold = float(threshold)
        self.weight_margin = float(weight_margin)
        self.weight_saliency = 1.0 - self.weight_margin
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.use_autocast = bool(use_autocast)
        self.autocast_dtype = autocast_dtype

        self.device = x_tensor.device
        self.model.eval()

    def _autocast_context(self):
        if self.use_autocast and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
        return nullcontext()

    def _call_explain(
        self,
        x_batch: torch.Tensor,
        target: torch.Tensor,
        create_graph: bool,
        detach: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            return self.explain_method(
                self.model,
                x_batch,
                self.normalize,
                target,
                create_graph=create_graph,
                detach=detach,
            )
        except TypeError:
            if create_graph:
                raise ValueError(
                    "Selected explain_method does not support differentiable saliency for PGD_sparse. "
                    "Use simple_gradient, input_gradient, integrated_gradients, or attention_grad."
                )
            return self.explain_method(self.model, x_batch, self.normalize, target)

    @staticmethod
    def _margin_loss(logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_true.numel() == 1:
            y_true = y_true.expand(logits.size(0))
        true_logits = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
        tmp = logits.clone()
        tmp.scatter_(1, y_true.unsqueeze(1), float("-inf"))
        max_other_logits = tmp.max(dim=1).values
        return (true_logits - max_other_logits).mean()

    def _joint_objective(
        self,
        x_adv: torch.Tensor,
        y_target: torch.Tensor,
        clean_saliency: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.model(self.normalize(x_adv))
        margin_loss = self._margin_loss(logits, y_target)

        adv_saliency, _ = self._call_explain(
            x_batch=x_adv,
            target=y_target,
            create_graph=True,
            detach=False,
        )
        saliency_loss = F.mse_loss(adv_saliency, clean_saliency)

        total_loss = -self.weight_margin * margin_loss + self.weight_saliency * saliency_loss
        return total_loss, margin_loss, saliency_loss, logits

    def attack(self) -> PGDSparseResult:
        y_target = self.y_true
        if y_target.numel() == 1 and self.x_tensor.size(0) > 1:
            y_target = y_target.expand(self.x_tensor.size(0))

        with self._autocast_context():
            clean_saliency, _ = self._call_explain(
                x_batch=self.x_tensor,
                target=y_target,
                create_graph=False,
                detach=True,
            )
        clean_saliency = clean_saliency.detach()

        dense_delta = torch.zeros_like(self.x_tensor, device=self.device)
        history: List[Dict[str, float]] = []
        first_success_iteration: Optional[int] = None

        for iteration in range(1, self.iterations + 1):
            dense_delta = dense_delta.detach().requires_grad_(True)
            delta_soft = torch.sigmoid(dense_delta)
            x_adv = torch.clamp(self.x_tensor + delta_soft, min=self.clip_min, max=self.clip_max)

            with self._autocast_context():
                total_loss, margin_loss, saliency_loss, logits = self._joint_objective(
                    x_adv=x_adv,
                    y_target=y_target,
                    clean_saliency=clean_saliency,
                )

            grad = torch.autograd.grad(total_loss, dense_delta, create_graph=False, retain_graph=False)[0]

            with torch.no_grad():
                dense_delta = dense_delta + self.step_size * grad.sign()

                pred = logits.argmax(dim=1)
                is_success = bool((pred != y_target).any().item())
                if first_success_iteration is None and is_success:
                    first_success_iteration = iteration

                history.append(
                    {
                        "iteration": float(iteration),
                        "margin_loss": float(margin_loss.detach().cpu().item()),
                        "saliency_loss": float(saliency_loss.detach().cpu().item()),
                        "weighted_fitness": float(total_loss.detach().cpu().item()),
                        "grad_l1": float(grad.detach().abs().mean().cpu().item()),
                        "first_success_iteration": first_success_iteration,
                    }
                )

        dense_delta = dense_delta.detach()
        delta_soft = torch.sigmoid(dense_delta)
        sparse_mask = (delta_soft > self.threshold).to(delta_soft.dtype)
        delta_sparse = sparse_mask * delta_soft
        x_adv_final = torch.clamp(self.x_tensor + delta_sparse, min=self.clip_min, max=self.clip_max)

        with self._autocast_context():
            final_total, final_margin, final_saliency, final_logits = self._joint_objective(
                x_adv=x_adv_final,
                y_target=y_target,
                clean_saliency=clean_saliency,
            )

        best_scores = {
            "margin_loss": float(final_margin.detach().cpu().item()),
            "saliency_loss": float(final_saliency.detach().cpu().item()),
            "weighted_fitness": float(final_total.detach().cpu().item()),
            "first_success_iteration": first_success_iteration,
            "adv_pred": int(final_logits.argmax(dim=1)[0].detach().cpu().item()),
            "l0_distance": int((sparse_mask > 0).sum().detach().cpu().item()),
            "sparse_ratio": float(sparse_mask.detach().float().mean().cpu().item()),
        }

        return PGDSparseResult(
            adv_chw=x_adv_final[0].detach(),
            dense_delta=dense_delta[0].detach(),
            delta_soft=delta_soft[0].detach(),
            delta_sparse=delta_sparse[0].detach(),
            sparse_mask=sparse_mask[0].detach(),
            best_scores=best_scores,
            history=history,
        )
