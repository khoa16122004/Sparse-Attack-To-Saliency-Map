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

    def _pick_best_candidate_idx(self, population, pop_weighted_fitness, pop_del_losses=None, pop_ins_losses=None):
        if pop_del_losses is not None and pop_ins_losses is not None:
            return self._pick_final_candidate_idx(population, pop_weighted_fitness, pop_del_losses, pop_ins_losses)
        success_indices = [i for i, sol in enumerate(population.population) if sol.is_adversarial]
        if success_indices:
            return min(success_indices, key=lambda idx: float(pop_weighted_fitness[idx].detach().cpu().item()))
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
        pop_margin_losses, pop_saliency_losses, pop_del_losses, pop_ins_losses, pop_logits = population.evaluate()    # calcuate fitenss    
        pop_weighted_fitness = self._compute_weighted_fitness(
            pop_margin_losses,
            pop_saliency_losses,
            pop_del_losses,
            pop_ins_losses,
        )
        pop_fitness = np.stack([pop_margin_losses.cpu().numpy(), pop_saliency_losses.cpu().numpy()], axis=1)
        selected_idxs, fronts, non_nominated_front = self.selection(pop_fitness)
        non_nominated_front_fitness = pop_fitness[non_nominated_front].copy()
        population = Population([population.population[i] for i in selected_idxs], self.params['fitness'])
        pop_margin_losses = pop_margin_losses[selected_idxs]
        pop_saliency_losses = pop_saliency_losses[selected_idxs]
        pop_del_losses = None if pop_del_losses is None else pop_del_losses[selected_idxs]
        pop_ins_losses = None if pop_ins_losses is None else pop_ins_losses[selected_idxs]
        pop_weighted_fitness = pop_weighted_fitness[selected_idxs]
        first_success_iteration = 0 if any(pi.is_adversarial for pi in population.population) else None

        best_candidate_id = self._pick_best_candidate_idx(population, pop_weighted_fitness, pop_del_losses, pop_ins_losses)
        best_candidate = population.population[best_candidate_id].copy()
        best_spearman = self._compute_history_spearman(best_candidate)
        best_scores = self._build_best_scores(
            pop_margin_losses,
            pop_saliency_losses,
            pop_weighted_fitness,
            best_candidate_id,
            first_success_iteration,
            pop_del_losses,
            pop_ins_losses,
            best_spearman,
        )
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
            off_margin_losses, off_saliency_losses, off_del_losses, off_ins_losses, off_logits = offpsrings.evaluate()
            off_weighted_fitness = self._compute_weighted_fitness(
                off_margin_losses,
                off_saliency_losses,
                off_del_losses,
                off_ins_losses,
            )
            pool_solutions = population.population + offpsrings.population
            pool_margin_losses = torch.cat([pop_margin_losses, off_margin_losses], dim=0)
            pool_saliency_losses = torch.cat([pop_saliency_losses, off_saliency_losses], dim=0)
            pool_del_losses = None if (pop_del_losses is None or off_del_losses is None) else torch.cat([pop_del_losses, off_del_losses], dim=0)
            pool_ins_losses = None if (pop_ins_losses is None or off_ins_losses is None) else torch.cat([pop_ins_losses, off_ins_losses], dim=0)
            pool_weighted_fitness = torch.cat([pop_weighted_fitness, off_weighted_fitness], dim=0)
            pool_fitness = np.stack([pool_margin_losses.cpu().numpy(), pool_saliency_losses.cpu().numpy()], axis=1)
            winner_idxs, fronts, non_nominated_front = self.selection(pool_fitness)
            non_nominated_front_fitness = pool_fitness[non_nominated_front]
            population = Population([pool_solutions[i] for i in winner_idxs], self.params['fitness'])
            pop_margin_losses = pool_margin_losses[winner_idxs]
            pop_saliency_losses = pool_saliency_losses[winner_idxs]
            pop_del_losses = None if pool_del_losses is None else pool_del_losses[winner_idxs]
            pop_ins_losses = None if pool_ins_losses is None else pool_ins_losses[winner_idxs]
            pop_weighted_fitness = pool_weighted_fitness[winner_idxs]

            if first_success_iteration is None and any(pi.is_adversarial for pi in population.population):
                first_success_iteration = it
            
            best_candidate_id = self._pick_best_candidate_idx(population, pop_weighted_fitness, pop_del_losses, pop_ins_losses)
            best_candidate = population.population[best_candidate_id].copy()

            best_spearman = self._compute_history_spearman(best_candidate)
            best_scores = self._build_best_scores(
                pop_margin_losses,
                pop_saliency_losses,
                pop_weighted_fitness,
                best_candidate_id,
                first_success_iteration,
                pop_del_losses,
                pop_ins_losses,
                best_spearman,
            )
            history.append(best_scores)
            # print(f"Iteration {it}: Best margin_loss={best_scores['margin_loss']:.4f}, Best saliency_loss={best_scores['saliency_loss']:.4f}")
        
        final_candidate_id = self._pick_final_candidate_idx(
            population,
            pop_weighted_fitness,
            pop_del_losses,
            pop_ins_losses,
        )
        best_candidate = population.population[final_candidate_id].copy()
        best_spearman = self._compute_history_spearman(best_candidate)
        best_scores = self._build_best_scores(
            pop_margin_losses,
            pop_saliency_losses,
            pop_weighted_fitness,
            final_candidate_id,
            first_success_iteration,
            pop_del_losses,
            pop_ins_losses,
            best_spearman,
        )

        return best_candidate.generate_adv_image(), best_candidate, best_scores, history, non_nominated_front_fitness
        
    def selection(self, fitnesess):
        pop_size = self.params["pop_size"]
        fronts = self.nds.do(fitnesess) # [ [id1, id2], [id3, id4] ,...]
        non_nominated_front = fronts[0]
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
        return selected_idxs, fronts, non_nominated_front

            
            