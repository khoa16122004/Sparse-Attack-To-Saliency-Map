import torch

from Solutions import Solution


def _value_distribution(zero_prob, device):
    return torch.tensor(
        [(1 - zero_prob) / 2, (1 - zero_prob) / 2, zero_prob],
        device=device,
    )


def _sample_values(n, zero_prob, device):
    probs = _value_distribution(zero_prob, device)
    base_values = torch.tensor([-1, 1, 0], device=device)
    return base_values[
        torch.multinomial(
            probs,
            n * 3,
            replacement=True,
        ).view(n, 3)
    ]


def build_pixel_sampling_probs(saliency_map, all_pixels, temperature=1.0, eps=1e-12):
    if saliency_map is None:
        return None

    saliency = saliency_map.detach().float().to(all_pixels.device)
    if saliency.dim() == 3 and saliency.size(0) == 1:
        saliency = saliency.squeeze(0)
    if saliency.dim() != 2:
        raise ValueError(f"Expected saliency map shape [H, W] or [1, H, W], got {tuple(saliency.shape)}")

    flat = saliency.reshape(-1)
    if flat.numel() != all_pixels.numel():
        raise ValueError(
            f"Saliency map has {flat.numel()} pixels but all_pixels has {all_pixels.numel()} entries"
        )

    temperature = max(float(temperature), 1e-6)
    probs = torch.softmax(flat / temperature, dim=0)
    probs = probs / (probs.sum() + eps)
    return probs


def _sample_pixels_without_replacement(candidates, n, candidate_probs=None):
    if n <= 0:
        return candidates[:0]
    if candidates.numel() == 0:
        raise ValueError("No candidate pixels available for sampling")

    n = min(int(n), int(candidates.numel()))
    device = candidates.device

    if candidate_probs is None:
        idx = torch.randperm(candidates.numel(), device=device)[:n]
        return candidates[idx]

    weights = candidate_probs.detach().float().to(device)
    weights = torch.clamp(weights, min=0.0)
    if torch.all(weights <= 0):
        idx = torch.randperm(candidates.numel(), device=device)[:n]
        return candidates[idx]

    idx = torch.multinomial(weights, n, replacement=False)
    return candidates[idx]


def init_population(pop_size, x_tensor, eps, p_size, zero_prob, all_pixels, pixel_probs=None):
    device = x_tensor.device
    solutions = []
    for _ in range(pop_size):
        if pixel_probs is None:
            pixels = torch.randperm(all_pixels.numel(), device=device)[:eps]
            pixels = all_pixels[pixels]
        else:
            pixels = _sample_pixels_without_replacement(all_pixels, eps, pixel_probs)

        values = _sample_values(eps, zero_prob, device)
        solutions.append(
            Solution(
                pixels,
                values,
                x_tensor.clone(),
                p_size,
            )
        )
    return solutions


def mutation(soln, pm, all_pixels, zero_prob, pixel_probs=None):
    device = soln.pixels.device

    eps = soln.pixels.numel()
    eps_it = max(int(eps * pm), 1)

    # Keep a subset from the current genome and replace the rest.
    keep = torch.randperm(eps, device=device)[: max(eps - eps_it, 0)]

    new_pixels = soln.pixels[keep]
    new_values = soln.values[keep]

    # Available positions that are not currently selected.
    mask = torch.ones(all_pixels.numel(), dtype=torch.bool, device=device)
    mask[soln.pixels] = False
    candidates = all_pixels[mask]

    n_replace = min(eps_it, candidates.numel())
    if n_replace > 0:
        candidate_probs = None if pixel_probs is None else pixel_probs[candidates]
        replace_pixels = _sample_pixels_without_replacement(candidates, n_replace, candidate_probs)
        replace_values = _sample_values(n_replace, zero_prob, device)

        soln.pixels = torch.cat((new_pixels, replace_pixels))
        soln.values = torch.cat((new_values, replace_values))
    else:
        soln.pixels = new_pixels
        soln.values = new_values


def crossover(soln1, soln2, pc, pixel_probs=None):
    device = soln1.pixels.device

    k = soln1.pixels.numel()
    l = max(int(k * pc), 1)

    offspring = soln1.copy()

    delta = (~torch.isin(soln2.pixels, soln1.pixels)).nonzero(as_tuple=True)[0]

    if delta.numel() > 0:
        n_take = min(l, delta.numel())
        if pixel_probs is None:
            pick = torch.randperm(delta.numel(), device=device)[:n_take]
            idx = delta[pick]
        else:
            delta_pixels = soln2.pixels[delta]
            delta_probs = pixel_probs[delta_pixels]
            picked_pixels = _sample_pixels_without_replacement(delta_pixels, n_take, delta_probs)
            picked_mask = torch.isin(soln2.pixels, picked_pixels)
            idx = picked_mask.nonzero(as_tuple=True)[0]

        offspring.pixels[idx] = soln2.pixels[idx]
        offspring.values[idx] = soln2.values[idx]

    return offspring


def generate_offspring(parents, pc, pm, all_pixels, zero_prob, pixel_probs=None):
    children = []

    for p1, p2 in parents:
        child = crossover(p1, p2, pc, pixel_probs=pixel_probs)
        mutation(child, pm, all_pixels, zero_prob, pixel_probs=pixel_probs)

        assert torch.unique(child.pixels).numel() == child.pixels.numel()

        children.append(child)

    return children






