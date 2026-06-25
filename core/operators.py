import torch

import numpy as np

def mutation(soln, pm, all_pixels, zero_prob):
    device = soln.pixels.device

    eps = soln.pixels.numel()
    eps_it = max(int(eps * pm), 1)

    # keep
    keep = torch.randperm(eps, device=device)[: eps - eps_it]

    new_pixels = soln.pixels[keep]
    new_values = soln.values[keep]

    # available pixels
    mask = torch.ones(all_pixels.numel(), dtype=torch.bool, device=device)
    mask[soln.pixels] = False
    candidates = all_pixels[mask]

    replace_pixels = candidates[
        torch.randperm(candidates.numel(), device=device)[:eps_it]
    ]

    probs = torch.tensor(
        [(1 - zero_prob) / 2, (1 - zero_prob) / 2, zero_prob],
        device=device,
    )

    replace_values = torch.tensor(
        [-1, 1, 0],
        device=device,
    )[
        torch.multinomial(
            probs,
            eps_it * 3,
            replacement=True,
        ).view(eps_it, 3)
    ]

    soln.pixels = torch.cat((new_pixels, replace_pixels))
    soln.values = torch.cat((new_values, replace_values))
    
def crossover(soln1, soln2, pc):
    device = soln1.pixels.device

    k = soln1.pixels.numel()
    l = max(int(k * pc), 1)

    offspring = soln1.copy()

    delta = (~torch.isin(soln2.pixels, soln1.pixels)).nonzero(as_tuple=True)[0]

    if delta.numel() > 0:
        idx = delta[
            torch.randperm(delta.numel(), device=device)[: min(l, delta.numel())]
        ]

        offspring.pixels[idx] = soln2.pixels[idx]
        offspring.values[idx] = soln2.values[idx]

    return offspring


def generate_offspring(parents, pc, pm, all_pixels, zero_prob):
    children = []

    for p1, p2 in parents:
        child = crossover(p1, p2, pc)
        mutation(child, pm, all_pixels, zero_prob)

        assert torch.unique(child.pixels).numel() == child.pixels.numel()

        children.append(child)

    return children






