import torch
from Solutions import Solution, Population
from weightedSUM_GA import Weighted_Sum_GA
from operators import build_pixel_sampling_probs, generate_offspring, init_population
from tqdm import tqdm
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np

class NSGAII(Weighted_Sum_GA):
    def __init__(self, params):
        super().__init__(params)
        self.nds = NonDominatedSorting()

    def _pick_best_candidate_idx(self, population, pop_saliency_losses):
        success_indices = [i for i, sol in enumerate(population.population) if sol.is_adversarial]
        if success_indices:
            return min(success_indices, key=lambda idx: float(pop_saliency_losses[idx].detach().cpu().item()))
        return min(range(len(population.population)), key=lambda idx: int(population.population[idx].l0.item()))
        
    def attack(self):
        init_solutions = init_population(
            pop_size=self.params["pop_size"],
            x_tensor=self.params["x_tensor"],
            eps=self.params["eps"],
            p_size=self.params["p_size"],
            zero_prob=self.params["zero_probability"],
            all_pixels=self.params["all_pixels"],
            pixel_probs=self.init_pixel_probs,
        )

        population = Population(init_solutions, self.params['fitness'])
        pop_margin_losses, pop_saliency_losses, pop_logits = population.evaluate()    # calcuate fitenss    
        pool_fitness = np.stack([pop_margin_losses.cpu().numpy(), pop_saliency_losses.cpu().numpy()], axis=1)
        selected_idxs, _ = self.selection(pool_fitness)
        population = Population([population.population[i] for i in selected_idxs], self.params['fitness'])
        pop_margin_losses = pop_margin_losses[selected_idxs]
        pop_saliency_losses = pop_saliency_losses[selected_idxs]
        first_success_iteration = 0 if any(pi.is_adversarial for pi in population.population) else None

        best_candidate_id = self._pick_best_candidate_idx(population, pop_saliency_losses)
        best_candidate = population.population[best_candidate_id].copy()
        best_scores = {
            'margin_loss': pop_margin_losses[best_candidate_id],
            'saliency_loss': pop_saliency_losses[best_candidate_id],
            'first_success_iteration': first_success_iteration,
        }
        history = [best_scores]

        for it in tqdm(range(1, self.params["iterations"])):            
            parent_indices = torch.randint(
                0,
                self.params["pop_size"],
                (self.params["pop_size"], 2),
                device=self.device,
            )
            parents = [
                (population.population[i1], population.population[i2])
                for i1, i2 in parent_indices
            ]
            
            offpsrings = self.generate_offpsrings(parents)
            offpsrings = Population(offpsrings, self.params['fitness'])
            off_margin_losses, off_saliency_losses, off_logits = offpsrings.evaluate()
            pool_solutions = population.population + offpsrings.population
            pool_margin_losses = torch.cat([pop_margin_losses, off_margin_losses], dim=0)
            pool_saliency_losses = torch.cat([pop_saliency_losses, off_saliency_losses], dim=0)
            pool_fitness = np.stack([pool_margin_losses.cpu().numpy(), pool_saliency_losses.cpu().numpy()], axis=1)
            winner_idxs, fronts = self.selection(pool_fitness)
            population = Population([pool_solutions[i] for i in winner_idxs], self.params['fitness'])
            pop_margin_losses = pool_margin_losses[winner_idxs]
            pop_saliency_losses = pool_saliency_losses[winner_idxs]

            if first_success_iteration is None and any(pi.is_adversarial for pi in population.population):
                first_success_iteration = it
            
            best_candidate_id = self._pick_best_candidate_idx(population, pop_saliency_losses)
            best_candidate = population.population[best_candidate_id].copy()
            
            best_scores = {
                'margin_loss': pop_margin_losses[best_candidate_id],
                'saliency_loss': pop_saliency_losses[best_candidate_id],
                'first_success_iteration': first_success_iteration,
            }
            history.append(best_scores)
            # print(f"Iteration {it}: Best margin_loss={best_scores['margin_loss']:.4f}, Best saliency_loss={best_scores['saliency_loss']:.4f}")
        
        return best_candidate.generate_adv_image(), best_candidate, best_scores, history
        
    def selection(self, fitnesess):
        pop_size = self.params["pop_size"]
        fronts = self.nds.do(fitnesess, n_stop_if_ranked=pop_size) # [ [id1, id2], [id3, id4] ,...]
        selected_idxs = []
        for k, front in enumerate(fronts):
            # front include indxs of fronts[k]
            crowding_of_front = self.calculating_crowding_distance(fitnesess[front])
            sorted_indices = np.argsort(-crowding_of_front)
            front_sorted = [front[i] for i in sorted_indices] # idxs sorted: [id2, id1]
            
            for idx in front_sorted:
                if len(selected_idxs) < pop_size:
                    selected_idxs.append(idx)
                else:
                    break
            if len(selected_idxs) >= pop_size:
                break
        return selected_idxs, fronts

            
            
    def calculating_crowding_distance(self, F):
        infinity = 1e+14

        n_points = F.shape[0]
        n_obj = F.shape[1]

        if n_points <= 2:
            return np.full(n_points, infinity)
        else:

            # sort each column and get index
            I = np.argsort(F, axis=0, kind='mergesort')

            # now really sort the whole array
            F = F[I, np.arange(n_obj)]

            # get the distance to the last element in sorted list and replace zeros with actual values
            dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) - np.concatenate([np.full((1, n_obj), -np.inf), F])

            index_dist_is_zero = np.where(dist == 0)

            dist_to_last = np.copy(dist)
            for i, j in zip(*index_dist_is_zero):
                dist_to_last[i, j] = dist_to_last[i - 1, j]

            dist_to_next = np.copy(dist)
            for i, j in reversed(list(zip(*index_dist_is_zero))):
                dist_to_next[i, j] = dist_to_next[i + 1, j]

            # normalize all the distances
            norm = np.max(F, axis=0) - np.min(F, axis=0)
            norm[norm == 0] = np.nan
            dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

            # if we divided by zero because all values in one columns are equal replace by none
            dist_to_last[np.isnan(dist_to_last)] = 0.0
            dist_to_next[np.isnan(dist_to_next)] = 0.0

            # sum up the distance to next and last and norm by objectives - also reorder from sorted list
            J = np.argsort(I, axis=0)
            crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        # replace infinity with a large number
        crowding[np.isinf(crowding)] = infinity
        return crowding