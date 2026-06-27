import torch
from Solutions import Solution, Population
from operators import build_pixel_sampling_probs, generate_offspring, init_population
from tqdm import tqdm


class Weighted_Sum_GA:
    def __init__(self, params):
        self.params = params
        self.device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.operator_strategy = params.get("operator_strategy", "uniform")
        self.saliency_temperature = params.get("saliency_temperature", 1.0)
        self.saliency_mix_start_alpha = float(params.get("saliency_mix_start_alpha", 0.1))
        self.saliency_mix_end_alpha = float(params.get("saliency_mix_end_alpha", 1.0))

        self.pixel_probs = None
        self.uniform_pixel_probs = None
        if self.operator_strategy == "saliency_guided":
            saliency_true = self.params["fitness"].saliency_true[0]
            self.pixel_probs = build_pixel_sampling_probs(
                saliency_true,
                self.params["all_pixels"],
                temperature=self.saliency_temperature,
            )
            self.uniform_pixel_probs = torch.full_like(
                self.pixel_probs,
                fill_value=1.0 / max(int(self.pixel_probs.numel()), 1),
            )

    def _generation_pixel_probs(self, generation_idx):
        if self.operator_strategy != "saliency_guided":
            return None

        # Keep initialization saliency-guided, then anneal operator sampling toward uniform.
        total_generations = max(int(self.params["iterations"]) - 1, 1)
        progress = float(generation_idx) / float(total_generations)
        progress = min(max(progress, 0.0), 1.0)

        alpha = self.saliency_mix_start_alpha + (
            self.saliency_mix_end_alpha - self.saliency_mix_start_alpha
        ) * progress
        alpha = min(max(alpha, 0.0), 1.0)

        mixed_probs = (1.0 - alpha) * self.pixel_probs + alpha * self.uniform_pixel_probs
        mixed_probs = mixed_probs / torch.clamp(mixed_probs.sum(), min=1e-12)
        return mixed_probs
    
    
    def attack(self):
        init_solutions = init_population(
            pop_size=self.params["pop_size"],
            x_tensor=self.params["x_tensor"],
            eps=self.params["eps"],
            p_size=self.params["p_size"],
            zero_prob=self.params["zero_probability"],
            all_pixels=self.params["all_pixels"],
            pixel_probs=self.pixel_probs,
        )

        population = Population(init_solutions, self.params['fitness'])
        pop_margin_losses, pop_saliency_losses, pop_logits = population.evaluate()    # calcuate fitenss    
        pop_weighted_fitness = self.params['w_margin'] * pop_margin_losses + self.params['w_saliency'] * pop_saliency_losses
        best_candidate_id = torch.argmin(pop_weighted_fitness)
        best_candidate = population.population[best_candidate_id].copy()
        best_scores = {
            'margin_loss': pop_margin_losses[best_candidate_id],
            'saliency_loss': pop_saliency_losses[best_candidate_id],
            'weighted_fitness': pop_weighted_fitness[best_candidate_id],
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
            
            generation_pixel_probs = self._generation_pixel_probs(it)
            offpsrings = self.generate_offpsrings(parents, pixel_probs=generation_pixel_probs)
            offpsrings = Population(offpsrings, self.params['fitness'])
            off_margin_losses, off_saliency_losses, off_logits = offpsrings.evaluate()  # calcuate fitenss
            off_weighted_fitness = self.params['w_margin'] * off_margin_losses + self.params['w_saliency'] * off_saliency_losses
            
            pool = population.population + offpsrings.population
            pool_margin_losses = torch.cat((pop_margin_losses, off_margin_losses), dim=0)
            pool_saliency_losses = torch.cat((pop_saliency_losses, off_saliency_losses), dim=0)
            pool_weighted_fitness = torch.cat((pop_weighted_fitness, off_weighted_fitness), dim=0)

            winner_idxs = self.tournament_selection(pool_weighted_fitness)
            population = Population([pool[i] for i in winner_idxs], self.params['fitness'])
            pop_margin_losses = pool_margin_losses[winner_idxs]
            pop_saliency_losses = pool_saliency_losses[winner_idxs]
            pop_weighted_fitness = pool_weighted_fitness[winner_idxs]
            
            best_candidate_id = torch.argmin(pop_weighted_fitness)
            best_candidate = population.population[best_candidate_id].copy()
            best_scores = {
                'margin_loss': pop_margin_losses[best_candidate_id],
                'saliency_loss': pop_saliency_losses[best_candidate_id],
                'weighted_fitness': pop_weighted_fitness[best_candidate_id],
            }
            history.append(best_scores)
            print(f"Iteration {it}: best weighted fitness = {best_scores['weighted_fitness']:.4f}, margin loss = {best_scores['margin_loss']:.4f}, saliency loss = {best_scores['saliency_loss']:.4f}")
        return best_candidate.generate_adv_image(), best_candidate, best_scores, history
            
            
            
            
            
    def tournament_selection(self, fitnesses):
        pool_idxs = torch.arange(len(fitnesses), device=self.device)
        selected_idxs = []
        
        for _ in range(2):
            # fshuffle the pool indices
            shuffled_idxs = pool_idxs[torch.randperm(len(pool_idxs), device=self.device)]
            for tournament in range(0, len(shuffled_idxs), 4):
                tournament_idxs = shuffled_idxs[tournament:tournament + 4]
                best_idx = tournament_idxs[torch.argmin(fitnesses[tournament_idxs])]
                selected_idxs.append(best_idx)
        
        return torch.tensor(selected_idxs, device=self.device)
            
    def generate_offpsrings(self, parents, pixel_probs=None):
        return generate_offspring(
            parents=parents,
            pc=self.params["pc"],
            pm=self.params["pm"],
            all_pixels=self.params["all_pixels"],
            zero_prob=self.params["zero_probability"],
            pixel_probs=pixel_probs,
        )
    
    
    

        

