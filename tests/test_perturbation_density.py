import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from perturbation_archive import PerturbationArchive, save_perturbation_density
from Solutions import Solution
from NSGAII import NSGAII
from weightedSUM_GA import Weighted_Sum_GA


def solution(pixels, values=None, clean=None):
    if clean is None:
        clean = torch.full((1, 3, 3, 4), 0.5)
    if values is None:
        values = [[1, 0, 0] for _ in pixels]
    return Solution(torch.tensor(pixels), torch.tensor(values), clean, 1.0)


def update(archive, solutions, scores):
    scores = torch.tensor(scores, dtype=torch.float32)
    archive.update(solutions, scores[:, 0], scores[:, 1])


class ArchiveTests(unittest.TestCase):
    def test_historical_front_duplicates_and_equal_objectives(self):
        archive = PerturbationArchive((3, 4))
        update(archive, [solution([0]), solution([1])], [[0, 3], [3, 0]])
        update(archive, [solution([2])], [[2, 2]])
        update(archive, [solution([3])], [[1, 1]])
        # A new point removes [2, 2], but neither historical extreme.
        self.assertEqual(len(archive.entries), 3)
        update(archive, [solution([3]), solution([4])], [[1, 1], [1, 1]])
        self.assertEqual(len(archive.entries), 4)
        update(archive, [solution([5])], [[4, 4]])
        self.assertEqual(len(archive.entries), 4)
        density, _, normalized = archive.density_maps(sigma=0)
        expected = torch.zeros(3, 4)
        expected.flatten()[torch.tensor([0, 1, 3, 4])] = 0.25
        torch.testing.assert_close(density, expected)
        torch.testing.assert_close(normalized, expected * 4)

    def test_actual_changes_clipping_and_genome_order(self):
        archive = PerturbationArchive((3, 4))
        clean = torch.zeros(1, 3, 3, 4)
        first = solution([0, 1, 2], [[1, 0, 0], [-1, 0, 0], [0, 0, 0]], clean)
        reordered = solution([2, 1, 0], [[0, 0, 0], [-1, 0, 0], [1, 0, 0]], clean)
        different_rgb = solution([1], [[0, -1, 0]], clean)
        update(archive, [first, reordered, different_rgb], [[1, 1]] * 3)
        self.assertEqual(len(archive.entries), 2)
        density, _, _ = archive.density_maps(0)
        self.assertEqual(density.sum().item(), 1)
        self.assertEqual(density.flatten()[1].item(), 1)

    def test_zero_change_is_empty_mask_not_missing_solution(self):
        archive = PerturbationArchive((3, 4))
        update(archive, [solution([0], [[0, 0, 0]]), solution([1])], [[0, 1], [1, 0]])
        density, _, _ = archive.density_maps(0)
        self.assertEqual(density.sum().item(), 0.5)

    def test_smoothing_and_invalid_inputs(self):
        archive = PerturbationArchive((3, 4))
        with self.assertRaises(ValueError):
            archive.density_maps()
        update(archive, [solution([5])], [[0, 0]])
        raw, smoothed, normalized = archive.density_maps(1)
        self.assertGreater(int((smoothed > 0).sum()), int((raw > 0).sum()))
        self.assertAlmostEqual(float(smoothed.sum()), 1.0, places=6)
        self.assertEqual(normalized.min().item(), 0)
        self.assertEqual(normalized.max().item(), 1)
        for sigma in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                archive.density_maps(sigma)
        with self.assertRaises(ValueError):
            update(archive, [solution([0])], [[float("nan"), 0]])

    def test_density_aggregates_pixel_frequency_in_4_by_4_grid(self):
        archive = PerturbationArchive((8, 8))
        clean = torch.full((1, 3, 8, 8), 0.5)
        update(
            archive,
            [solution([0], clean=clean), solution([18], clean=clean)],
            [[0, 1], [1, 0]],
        )

        grid = archive.grid_density_map()

        self.assertEqual(tuple(grid.shape), (4, 4))
        self.assertAlmostEqual(grid[0, 0].item(), 0.125)
        self.assertAlmostEqual(grid[1, 1].item(), 0.125)
        self.assertEqual(grid.sum().item(), 0.25)
        density, _, _ = archive.density_maps(sigma=0)
        self.assertEqual(tuple(density.shape), (8, 8))

    def test_density_uses_linear_minmax_normalization(self):
        archive = PerturbationArchive((32, 32))
        clean = torch.full((1, 3, 32, 32), 0.5)
        update(archive, [solution([330], clean=clean)], [[0, 0]])
        _, smoothed, normalized = archive.density_maps(sigma=1)
        expected = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
        torch.testing.assert_close(normalized, expected)
        self.assertTrue(((normalized > 0) & (normalized < 1)).any())

    def test_constant_map_and_saved_outputs(self):
        archive = PerturbationArchive((3, 4))
        update(archive, [solution([0], [[0, 0, 0]])], [[0, 0]])
        clean = torch.full((3, 3, 4), 0.5)

        def save_map(tensor, path):
            Image.fromarray((tensor.numpy() * 255).astype(np.uint8)).save(path)

        with tempfile.TemporaryDirectory() as directory:
            metadata = save_perturbation_density(
                archive, clean, Path(directory) / "density", save_map
            )
            for path in metadata["paths"].values():
                self.assertTrue(Path(path).is_file())
            with open(metadata["paths"]["metadata"], encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), metadata)
            self.assertEqual(metadata["sigma"], 1.0)
            self.assertNotIn("gamma", metadata)
            data = torch.load(metadata["paths"]["data"], weights_only=True)
            self.assertEqual(set(data), {
                "density", "grid_density", "smoothed_density",
                "normalized_density", "archive"
            })
            self.assertEqual(data["normalized_density"].sum().item(), 0)
            self.assertTrue(all(not entry["values"].requires_grad for entry in data["archive"]))
            with Image.open(metadata["paths"]["overlay"]) as image:
                self.assertTrue((np.asarray(image) == 128).all())


class RecordingFitness:
    def __init__(self):
        self.y_true = torch.tensor([0])
        self.saliency_true = torch.ones(1, 3, 4)
        self.images = []

    def benchmark(self, images):
        images = images.detach()
        self.images.extend(images.clone())
        score = images.flatten(1).sum(dim=1)
        logits = torch.stack((score, -score), dim=1)
        return score, -score, logits


class AttackIntegrationTests(unittest.TestCase):
    def test_both_algorithms_archive_every_evaluation_without_changing_rng(self):
        for algorithm in [Weighted_Sum_GA, NSGAII]:
            with self.subTest(algorithm=algorithm.__name__):
                fitness = RecordingFitness()
                params = {
                    "x_tensor": torch.full((1, 3, 3, 4), 0.5),
                    "fitness": fitness, "pop_size": 4, "iterations": 3,
                    "eps": 2, "p_size": 1.0, "zero_probability": 0.3,
                    "all_pixels": torch.arange(12), "w_margin": 0.5,
                    "w_saliency": 0.5, "pc": 0.4, "pm": 0.1, "device": "cpu",
                }
                torch.manual_seed(7)
                attacker = algorithm(params)
                original_archive = attacker.pareto_archive
                output = attacker.attack()
                self.assertIsNot(attacker.pareto_archive, original_archive)
                self.assertEqual(len(output), 5)
                self.assertEqual(len(fitness.images), 12)
                unique = {image.numpy().tobytes(): image for image in fitness.images}
                expected = torch.stack([
                    (image != params["x_tensor"][0]).any(dim=0).float()
                    for image in unique.values()
                ]).mean(dim=0)
                density, _, _ = attacker.pareto_archive.density_maps(0)
                self.assertEqual(len(attacker.pareto_archive.entries), len(unique))
                torch.testing.assert_close(density, expected)
                rng_with_archive = torch.get_rng_state()
                torch.manual_seed(7)
                baseline = algorithm(params)
                with patch.object(PerturbationArchive, "update"):
                    baseline_output = baseline.attack()
                torch.testing.assert_close(output[0], baseline_output[0])
                torch.testing.assert_close(rng_with_archive, torch.get_rng_state())
                self.assertEqual(output[3], baseline_output[3])
                np.testing.assert_array_equal(output[4], baseline_output[4])


if __name__ == "__main__":
    unittest.main()
