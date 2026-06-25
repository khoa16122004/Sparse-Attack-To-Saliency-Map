from icecream import Any
import numpy as np
from copy import deepcopy
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
        print("imgs_adv shape: ", imgs_adv.shape)
        margin_losses, saliency_losses, logits =  self.fitness.benchmark(imgs_adv)
        for idx, pi in enumerate(self.population):
            pi.margin_loss = margin_losses[idx]
            pi.saliency_loss = saliency_losses[idx]
            pi.l0 = pi.l0_distance(imgs_adv[idx])
            pi.pred_label = logits[idx].argmax().item()
            y_true_item = self.fitness.y_true[0].item() if self.fitness.y_true.numel() == 1 else self.fitness.y_true[idx].item()
            pi.is_adversarial = pi.pred_label != y_true_item
            
        return margin_losses, saliency_losses, logits  
        

            




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
        return deepcopy(self)

    def euc_distance(self, img):
        return np.sum((img - self.x.copy()) ** 2)

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

        x_adv_[rows, cols] += self.values * self.p_size
        x_adv_ = x_adv_.clamp_(0.0, 1.0)
        x_adv = x_adv_.permute(2, 0, 1) # 3 x w x h

        return x_adv
    
