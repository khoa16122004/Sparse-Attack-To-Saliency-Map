from typing import Any
import numpy as np
from operator import attrgetter
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import torch




class Population:
    def __init__(self, solutions: list, fitness_function: Any):
        self.population = solutions
        self.fronts = None
        self.fitness = fitness_function

    def evaluate(self):
        imgs_adv = torch.stack(
            [pi.generate_adv_image() for pi in self.population],
            dim=0
        )
        # print("imgs_adv shape: ", imgs_adv.shape)
        outputs = self.fitness.benchmark(imgs_adv)
        if len(outputs) == 3:
            margin_losses, saliency_losses, logits = outputs
            del_losses = None
            ins_losses = None
        elif len(outputs) == 4:
            margin_losses, del_losses, ins_losses, logits = outputs
            saliency_losses = del_losses + ins_losses
        else:
            raise ValueError("fitness.benchmark must return (margin, saliency, logits) or (margin, del, ins, logits)")

        for idx, pi in enumerate(self.population):
            pi.margin_loss = margin_losses[idx]
            pi.saliency_loss = saliency_losses[idx]
            pi.del_loss = None if del_losses is None else del_losses[idx]
            pi.ins_loss = None if ins_losses is None else ins_losses[idx]
            pi.l0 = pi.l0_distance(imgs_adv[idx])
            pi.pred_label = logits[idx].argmax().item()
            y_true_item = self.fitness.y_true[0].item() if self.fitness.y_true.numel() == 1 else self.fitness.y_true[idx].item()
            pi.is_adversarial = pi.pred_label != y_true_item
            
        return margin_losses, saliency_losses, del_losses, ins_losses, logits  
        

            




class Solution:
    def __init__(self, pixels, values, x, p_size):
        self.pixels = pixels  # list of Integers
        self.values = values 
        self.x = x  # (w x w x 3)
        self.fitnesses = []
        self.is_adversarial = None
        # x is expected to be (1, 3, H, W); use W for linear pixel indexing.
        self.w = x.shape[-1]
        self.delta = len(self.pixels)
        self.domination_count = None
        self.dominated_solutions = None
        self.rank = None
        self.crowding_distance = None

        self.loss = None
        self.pred_label = -1
        self.p_size = p_size

    def copy(self):
        # Avoid Python deepcopy on tensors that may not be graph leaves.
        pixels = self.pixels.clone() if torch.is_tensor(self.pixels) else np.array(self.pixels, copy=True)
        values = self.values.clone() if torch.is_tensor(self.values) else np.array(self.values, copy=True)
        x = self.x.clone() if torch.is_tensor(self.x) else np.array(self.x, copy=True)

        cloned = Solution(pixels=pixels, values=values, x=x, p_size=self.p_size)
        cloned.fitnesses = list(self.fitnesses)
        cloned.is_adversarial = self.is_adversarial
        cloned.w = self.w
        cloned.delta = self.delta
        cloned.domination_count = self.domination_count
        cloned.dominated_solutions = self.dominated_solutions
        cloned.rank = self.rank
        cloned.crowding_distance = self.crowding_distance
        cloned.loss = self.loss
        cloned.pred_label = self.pred_label

        for name in ("margin_loss", "saliency_loss", "del_loss", "ins_loss", "l0"):
            if hasattr(self, name):
                value = getattr(self, name)
                if torch.is_tensor(value):
                    value = value.detach().clone()
                cloned.__dict__[name] = value

        return cloned

    def euc_distance(self, img):
        if torch.is_tensor(self.x):
            return torch.sum((img - self.x) ** 2)
        return np.sum((img - np.array(self.x, copy=True)) ** 2)

    def l0_distance(self, img):
        base = self.x.squeeze(0)
        # Count pixels where at least one channel changed.
        return (img != base).any(dim=0).sum()

    def generate_adv_image(self):
        x_adv = self.x.clone().squeeze(0) # 3 x w x h
        x_adv_ = x_adv.permute(1, 2, 0) # w x h x 3
        # self.value: 50 x 3
        rows = self.pixels // self.w
        cols = self.pixels % self.w

        # x_adv_[rows, cols] += self.values * self.p_size
        x_adv_[rows, cols] -= self.values * self.p_size
        x_adv_ = x_adv_.clamp_(0.0, 1.0)
        x_adv = x_adv_.permute(2, 0, 1) # 3 x w x h

        return x_adv
    
