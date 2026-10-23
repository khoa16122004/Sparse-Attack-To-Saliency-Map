import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter


DENSITY_GRID_SIZE = (4, 4)


class PerturbationArchive:
    """Unique realized perturbations, minimizing the two original objectives."""

    def __init__(self, image_shape):
        self.image_shape = tuple(image_shape)
        self.entries = {}

    @torch.no_grad()
    def update(self, solutions, margin_losses, saliency_losses):
        objectives = torch.stack(
            (margin_losses.detach(), saliency_losses.detach()), dim=1
        ).cpu().double().numpy()
        if objectives.shape != (len(solutions), 2) or not np.isfinite(objectives).all():
            raise ValueError("Archive requires two finite objectives per solution.")

        for solution, scores in zip(solutions, objectives):
            previous_scores = (
                np.stack([entry["objectives"].numpy() for entry in self.entries.values()])
                if self.entries else np.empty((0, 2))
            )
            dominated = np.all(previous_scores <= scores, axis=1) & np.any(
                previous_scores < scores, axis=1
            )
            if dominated.any():
                continue

            adv = solution.generate_adv_image().detach()
            clean = solution.x.detach().squeeze(0)
            if tuple(adv.shape[-2:]) != self.image_shape:
                raise ValueError("Archive image shape does not match the solution.")
            pixels = (adv != clean).any(dim=0).flatten().nonzero().flatten().cpu()
            values = adv.permute(1, 2, 0).reshape(-1, 3)[
                pixels.to(adv.device)
            ].cpu().clone()
            # Identity is the actual RGB perturbation, independent of genome order.
            key = (pixels.numpy().tobytes(), values.numpy().tobytes())
            if key in self.entries:
                continue

            removed = np.all(scores <= previous_scores, axis=1) & np.any(
                scores < previous_scores, axis=1
            )
            for old_key, remove in zip(list(self.entries), removed):
                if remove:
                    del self.entries[old_key]
            self.entries[key] = {
                "pixels": pixels,
                "values": values,
                "objectives": torch.from_numpy(scores.copy()),
            }

    def grid_density_map(self, grid_size=DENSITY_GRID_SIZE):
        if not self.entries:
            raise ValueError("Cannot generate a density map from an empty archive.")
        height, width = self.image_shape
        grid_height = min(grid_size[0], height)
        grid_width = min(grid_size[1], width)
        cell_count = grid_height * grid_width
        pixel_indices = torch.arange(height * width)
        rows = pixel_indices // width
        columns = pixel_indices % width
        cell_indices = (
            (rows * grid_height // height) * grid_width
            + columns * grid_width // width
        )
        cell_areas = torch.bincount(cell_indices, minlength=cell_count).float()
        counts = torch.zeros(cell_count, dtype=torch.float32)
        for entry in self.entries.values():
            pixels = entry["pixels"]
            pixel_rows = pixels // width
            pixel_columns = pixels % width
            cells = (
                (pixel_rows * grid_height // height) * grid_width
                + pixel_columns * grid_width // width
            )
            counts += torch.bincount(cells, minlength=cell_count).float()
        return (counts / len(self.entries) / cell_areas).reshape(
            grid_height, grid_width
        )

    def density_maps(self, sigma=1.0, gamma=2.0):
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("Density sigma must be finite and non-negative.")
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("Density gamma must be finite and positive.")
        grid_density = self.grid_density_map()
        density = torch.nn.functional.interpolate(
            grid_density.unsqueeze(0).unsqueeze(0),
            size=self.image_shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        smoothed = torch.from_numpy(gaussian_filter(density.numpy(), sigma=sigma))
        low, high = smoothed.min(), smoothed.max()
        normalized = (
            (smoothed - low) / (high - low)
            if high > low else torch.zeros_like(smoothed)
        )
        return density, smoothed, normalized.pow(gamma)


def save_perturbation_density(archive, clean_chw, output_prefix, save_map, sigma=1.0, gamma=2.0):
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw_map": f"{prefix}_raw.png",
        "map": f"{prefix}.png",
        "overlay": f"{prefix}_overlay.png",
        "data": f"{prefix}.pt",
        "metadata": f"{prefix}.json",
    }
    density, smoothed, normalized = archive.density_maps(sigma, gamma)
    grid_density = archive.grid_density_map()
    save_map(density, paths["raw_map"])
    save_map(normalized, paths["map"])

    heat = torch.stack([
        (3 * normalized).clamp(0, 1),
        (3 * normalized - 1).clamp(0, 1),
        (3 * normalized - 2).clamp(0, 1),
    ])
    clean = clean_chw.detach().cpu().float()
    alpha = 0.5 * normalized.unsqueeze(0)
    overlay = ((1 - alpha) * clean + alpha * heat).clamp(0, 1)
    Image.fromarray(
        (overlay.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    ).save(paths["overlay"])
    torch.save({
        "density": density,
        "grid_density": grid_density,
        "smoothed_density": smoothed,
        "normalized_density": normalized,
        "archive": list(archive.entries.values()),
    }, paths["data"])
    metadata = {
        "archive_size": len(archive.entries),
        "image_shape": list(archive.image_shape),
        "sigma": sigma,
        "gamma": gamma,
        "grid_size": list(DENSITY_GRID_SIZE),
        "grid_value": "mean changed-pixel frequency within each cell",
        "objectives": ["margin_loss", "saliency_loss"],
        "objective_directions": ["min", "min"],
        "mask_definition": "any RGB channel differs from clean after clamping",
        "uniqueness": "actual changed pixel indices and RGB values",
        "paths": paths,
    }
    with open(paths["metadata"], "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata
