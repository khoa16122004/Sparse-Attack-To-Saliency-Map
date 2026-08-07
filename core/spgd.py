import copy
import math
import torch
import torch.nn as nn


class SaliencySparsePGD:
    def __init__(
        self,
        model,
        normalize,
        explain_method,
        epsilon=8.0 / 255.0,
        k=50,
        t=40,
        alpha=1.0 / 255.0,
        beta=0.25,
        random_start=True,
        attack_mode="pixel",
        w_margin=1.0,
        w_saliency=1.0,
        tau=0.5,
        sparsity_ratio=None,
        fixed_mask_location=True,
        zero_grad_patience=3,
        zero_grad_jitter=1e-2,
        use_softplus_surrogate=True,
        softplus_beta=10.0,
        debug_grad=False,
    ):
        self.model = model
        self.normalize = normalize
        self.explain_method = explain_method
        self.epsilon = float(epsilon)
        self.k = int(k)
        self.t = int(t)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.random_start = bool(random_start)
        self.attack_mode = attack_mode
        self.w_margin = float(w_margin)
        self.w_saliency = float(w_saliency)
        self.tau = float(tau)
        self.sparsity_ratio = sparsity_ratio
        self.fixed_mask_location = bool(fixed_mask_location)
        self.zero_grad_patience = int(zero_grad_patience)
        self.zero_grad_jitter = float(zero_grad_jitter)
        self.use_softplus_surrogate = bool(use_softplus_surrogate)
        self.softplus_beta = float(softplus_beta)
        self.debug_grad = bool(debug_grad)

        self.saliency_model = self._build_saliency_model()

        if self.attack_mode not in {"pixel", "feature"}:
            raise ValueError("attack_mode must be 'pixel' or 'feature'")
        if not 0.0 < self.tau < 1.0:
            raise ValueError("tau must be in (0, 1)")
        if self.sparsity_ratio is not None:
            self.sparsity_ratio = float(self.sparsity_ratio)
            if not 0.0 < self.sparsity_ratio <= 1.0:
                raise ValueError("sparsity_ratio must be in (0, 1]")
        if self.zero_grad_patience < 1:
            raise ValueError("zero_grad_patience must be >= 1")
        if self.zero_grad_jitter < 0.0:
            raise ValueError("zero_grad_jitter must be >= 0")
        if self.softplus_beta <= 0.0:
            raise ValueError("softplus_beta must be > 0")

    @staticmethod
    def _replace_relu_with_softplus(module, beta):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.Softplus(beta=beta))
            else:
                SaliencySparsePGD._replace_relu_with_softplus(child, beta)

    def _build_saliency_model(self):
        if not self.use_softplus_surrogate:
            return self.model

        surrogate = copy.deepcopy(self.model)
        self._replace_relu_with_softplus(surrogate, self.softplus_beta)
        surrogate.eval()
        # Attack optimizes input/mask only; model weights should stay frozen.
        for p in surrogate.parameters():
            p.requires_grad_(False)
        return surrogate

    @staticmethod
    def _prepare_targets(logits, y_true):
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, device=logits.device)
        y_true = y_true.to(device=logits.device, dtype=torch.long).view(-1)
        if y_true.numel() == 1 and logits.size(0) > 1:
            y_true = y_true.expand(logits.size(0))
        if y_true.numel() != logits.size(0):
            raise ValueError(
                f"y_true has {y_true.numel()} elements, expected batch size {logits.size(0)}"
            )
        return y_true

    @staticmethod
    def _margin_loss(logits, y_true):
        y_true = SaliencySparsePGD._prepare_targets(logits, y_true)
        true_logits = logits.gather(1, y_true.view(-1, 1)).squeeze(1)
        others = logits.clone()
        others.scatter_(1, y_true.view(-1, 1), float("-inf"))
        other_logits = others.max(dim=1).values
        return -(true_logits - other_logits)

    @staticmethod
    def _margin_objective(logits, y_true):
        y_true = SaliencySparsePGD._prepare_targets(logits, y_true)
        true_logits = logits.gather(1, y_true.view(-1, 1)).squeeze(1)
        others = logits.clone()
        others.scatter_(1, y_true.view(-1, 1), float("-inf"))
        other_logits = others.max(dim=1).values
        return true_logits - other_logits

    @staticmethod
    def _mse_per_sample(a, b):
        a = a.flatten(start_dim=1)
        b = b.flatten(start_dim=1)
        if a.size(0) == 1 and b.size(0) > 1:
            a = a.expand(b.size(0), -1)
        return ((a - b) ** 2).mean(dim=1)

    def _call_explain_method(self, x_tensor, y_true, detach):
        # Prefer backprop-capable APIs from explain_method_backprop.
        try:
            saliency, logits = self.explain_method(
                self.saliency_model,
                x_tensor,
                self.normalize,
                target_class=y_true,
                create_graph=not detach,
                detach=detach,
            )
            if not detach and not saliency.requires_grad:
                raise RuntimeError(
                    "Saliency map is not differentiable (requires_grad=False). "
                    "Use explain_method_backprop and ensure detach=False with create_graph=True."
                )
            return saliency, logits
        except TypeError as exc:
            saliency, logits = self.explain_method(
                self.saliency_model,
                x_tensor,
                self.normalize,
                y_true,
            )
            if detach:
                return saliency.detach(), logits.detach()
            if not saliency.requires_grad:
                raise RuntimeError(
                    "Fallback explain_method API returned non-differentiable saliency. "
                    "Please pass a backprop-capable method from explain_method_backprop."
                ) from exc
            return saliency, logits

    def _initialize_state(self, x):
        if self.random_start:
            perturb = x.new_empty(x.size()).uniform_(-self.epsilon, self.epsilon)
        else:
            perturb = x.new_zeros(x.size())
        perturb = torch.min(torch.max(perturb, -x), 1 - x)
        return perturb

    def _expand_to_input(self, t, x):
        if t.size(1) == 1 and x.size(1) > 1:
            return t.expand(-1, x.size(1), -1, -1)
        return t

    def _resolve_k(self, x):
        if self.attack_mode == "pixel":
            total = x.size(2) * x.size(3)
        else:
            total = x[0].numel()
        if self.sparsity_ratio is not None:
            return max(1, min(int(round(self.sparsity_ratio * total)), total))
        return max(1, min(int(self.k), total))

    @staticmethod
    def _normalize_grad_l1(grad):
        return grad / (1e-10 + grad.abs().sum(dim=(1, 2, 3), keepdim=True))

    @staticmethod
    def _hard_topk_ste(prob, k):
        flat = prob.view(prob.size(0), -1)
        _, idx = torch.topk(flat, k=k, dim=1, largest=True, sorted=False)
        hard = torch.zeros_like(flat).scatter_(1, idx, 1.0).view_as(prob)
        # Forward uses hard top-k mask; backward follows soft probabilities.
        st = prob + (hard - prob).detach()
        return st, hard

    def _topk_support_mask(self, y, x, k):
        if self.attack_mode == "pixel":
            lb = -x
            ub = 1.0 - x
            p1 = y.pow(2).sum(dim=1)
            p2 = torch.minimum(torch.minimum(ub - y, y - lb), torch.zeros_like(y)).pow(2).sum(dim=1)
            score = p1 - p2
            flat = score.view(score.size(0), -1)
            _, idx = torch.topk(flat, k=k, dim=1, largest=True, sorted=False)
            return torch.zeros_like(flat).scatter_(1, idx, 1.0).view(score.size(0), 1, score.size(1), score.size(2))

        flat = y.view(y.size(0), -1)
        score = flat.pow(2)
        _, idx = torch.topk(score, k=k, dim=1, largest=True, sorted=False)
        return torch.zeros_like(flat).scatter_(1, idx, 1.0).view_as(y)

    def _project_l0_box(self, y, x, k, fixed_support=None):
        # Project perturbation y to satisfy: <=k changed pixels and box bounds around x.
        lb = -x
        ub = 1.0 - x

        y_clamped = torch.min(torch.max(y, lb), ub)

        if fixed_support is not None:
            return y_clamped * fixed_support

        if self.attack_mode == "pixel":
            mask = self._topk_support_mask(y, x, k)
            return y_clamped * mask

        # feature mode: sparsity across all channel-spatial entries
        flat = y_clamped.view(y_clamped.size(0), -1)
        mask = self._topk_support_mask(y, x, k).view(flat.size(0), -1)
        return (flat * mask).view_as(y_clamped)

    @staticmethod
    def _ste_clip(x, x_min=0.0, x_max=1.0):
        x_clamped = torch.clamp(x, x_min, x_max)
        # Forward uses clamped value; backward uses identity gradient.
        return x + (x_clamped - x).detach()

    def attack(self, x, y_true, saliency_ref=None, return_history=True):
        if self.t <= 0:
            x_adv = x.clone()
            return (x_adv, []) if return_history else x_adv

        training = self.model.training
        self.model.eval()

        perturb = self._initialize_state(x)
        k_eff = self._resolve_k(x)

        if saliency_ref is None:
            saliency_ref, _ = self._call_explain_method(x, y_true, detach=True)
        saliency_ref = saliency_ref.to(device=x.device, dtype=x.dtype)
        if saliency_ref.dim() == 2:
            saliency_ref = saliency_ref.unsqueeze(0)
        # Keep reference saliency graph-free so backward is independent per PGD step.
        saliency_ref = saliency_ref.detach()

        history = []
        best_score = torch.full((x.size(0),), float("-inf"), device=x.device)
        best_perturb = perturb.detach().clone()
        zero_grad_streak = 0
        fixed_support = None
        mask_logits = None
        best_mask_logits = None
        if self.fixed_mask_location:
            fixed_support = self._topk_support_mask(perturb.detach(), x, k_eff)
        else:
            init_support = self._topk_support_mask(perturb.detach(), x, k_eff)
            mask_logits = torch.where(
                init_support > 0.5,
                torch.full_like(init_support, 2.0),
                torch.full_like(init_support, -2.0),
            )
            best_mask_logits = mask_logits.detach().clone()

        for _ in range(self.t):
            perturb = perturb.detach().requires_grad_(True)
            if mask_logits is not None:
                mask_logits = mask_logits.detach().requires_grad_(True)

            if mask_logits is None:
                sparse_perturb = self._project_l0_box(perturb, x, k_eff, fixed_support=fixed_support)
                sparse_perturb = torch.clamp(sparse_perturb, -self.epsilon, self.epsilon)
                hard_support = fixed_support
            else:
                mask_prob = torch.sigmoid(mask_logits)
                mask_st, hard_support = self._hard_topk_ste(mask_prob, k_eff)
                sparse_perturb = torch.clamp(perturb, -self.epsilon, self.epsilon) * self._expand_to_input(mask_st, x)
                sparse_perturb = torch.min(torch.max(sparse_perturb, -x), 1.0 - x)

            x_adv_raw = x + sparse_perturb
            x_adv = self._ste_clip(x_adv_raw, 0.0, 1.0)

            saliency_adv, _ = self._call_explain_method(x_adv, y_true, detach=False)
            logits_adv = self.model(self.normalize(x_adv))
            margin = self._margin_loss(logits_adv, y_true)
            saliency_mse = self._mse_per_sample(saliency_ref, saliency_adv)
            saliency_objective = saliency_mse
            total = self.w_margin * margin + self.w_saliency * saliency_objective
            total_mean = total.mean()

            if self.debug_grad:
                margin_term = (self.w_margin * margin).mean()
                saliency_term = (self.w_saliency * saliency_objective).mean()

                grad_margin = torch.autograd.grad(
                    margin_term,
                    perturb,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                grad_saliency = torch.autograd.grad(
                    saliency_term,
                    perturb,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )[0]

                grad_mask_dbg = None
                if mask_logits is not None:
                    grad_mask_dbg = torch.autograd.grad(
                        total_mean,
                        mask_logits,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )[0]

                def _safe_norm(t):
                    if t is None:
                        return 0.0
                    return float(t.detach().norm().cpu().item())

                if mask_logits is None:
                    grad_mask_info = "N/A(hard top-k support)"
                else:
                    grad_mask_info = _safe_norm(grad_mask_dbg)

                print(
                    "Margin loss:",
                    float(margin.mean().detach().cpu().item()),
                    "Saliency MSE:",
                    float(saliency_mse.mean().detach().cpu().item()),
                    "Saliency objective(MSE):",
                    float(saliency_objective.mean().detach().cpu().item()),
                    "Weighted fitness:",
                    float(total_mean.detach().cpu().item()),
                    "| grad_margin(delta)=",
                    _safe_norm(grad_margin),
                    "| grad_saliency(delta)=",
                    _safe_norm(grad_saliency),
                    "| grad_mask=",
                    grad_mask_info,
                    "| saliency_requires_grad=",
                    bool(saliency_adv.requires_grad),
                )

            grad_targets = [perturb] if mask_logits is None else [perturb, mask_logits]
            grads = torch.autograd.grad(
                total_mean,
                grad_targets,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            grad_perturb = grads[0]
            grad_mask = grads[1] if mask_logits is not None else None

            with torch.no_grad():
                grad_perturb_norm = float(grad_perturb.detach().abs().mean().cpu().item())
                if grad_perturb_norm <= 1e-12:
                    zero_grad_streak += 1
                else:
                    zero_grad_streak = 0

                improve = total > best_score
                if improve.any():
                    best_score[improve] = total[improve]
                    best_perturb[improve] = perturb.detach()[improve]
                    if best_mask_logits is not None:
                        best_mask_logits[improve] = mask_logits.detach()[improve]

                grad_update = self._normalize_grad_l1(grad_perturb)
                perturb = perturb + self.alpha * grad_update

                if mask_logits is None:
                    perturb = self._project_l0_box(perturb, x, k_eff, fixed_support=fixed_support)
                    perturb = torch.clamp(perturb, -self.epsilon, self.epsilon)
                else:
                    mask_logits = mask_logits + self.beta * self._normalize_grad_l1(grad_mask)
                    perturb = torch.clamp(perturb, -self.epsilon, self.epsilon)
                    perturb = torch.min(torch.max(perturb, -x), 1.0 - x)

                if zero_grad_streak >= self.zero_grad_patience and self.zero_grad_jitter > 0.0:
                    perturb = perturb + self.zero_grad_jitter * torch.randn_like(perturb)
                    zero_grad_streak = 0

                if return_history:
                    margin_obj = self._margin_objective(logits_adv, y_true)
                    history.append(
                        {
                            "margin_objective": float(margin_obj.mean().detach().cpu().item()),
                            "margin_loss": float(margin.mean().detach().cpu().item()),
                            "saliency_mse": float(saliency_mse.mean().detach().cpu().item()),
                            "saliency_objective": float(saliency_objective.mean().detach().cpu().item()),
                            "weighted_fitness": float(total.mean().detach().cpu().item()),
                        }
                    )

        with torch.no_grad():
            if best_mask_logits is None:
                sparse_perturb = self._project_l0_box(best_perturb, x, k_eff, fixed_support=fixed_support)
                sparse_perturb = torch.clamp(sparse_perturb, -self.epsilon, self.epsilon)
            else:
                best_mask_prob = torch.sigmoid(best_mask_logits)
                _, best_hard_support = self._hard_topk_ste(best_mask_prob, k_eff)
                sparse_perturb = torch.clamp(best_perturb, -self.epsilon, self.epsilon) * self._expand_to_input(best_hard_support, x)
                sparse_perturb = torch.min(torch.max(sparse_perturb, -x), 1.0 - x)
            x_adv_best = torch.clamp(x + sparse_perturb, 0.0, 1.0)

        if training:
            self.model.train()

        if return_history:
            return x_adv_best.detach(), history
        return x_adv_best.detach()

    def __call__(self, x, y_true, saliency_ref=None):
        return self.attack(x, y_true, saliency_ref=saliency_ref, return_history=False)
