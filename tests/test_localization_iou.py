import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluate_script.generate_imagenet_masks import best_mask, load_prompts
from evaluate_script.run_localization_iou import (
    align_target, binary_iou, load_clean_map, save_reports, top_k_mask,
)


class MaskGenerationTests(unittest.TestCase):
    def test_prompts_default_and_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synsets = root / "synsets.txt"
            synsets.write_text("n001 cat, domestic cat\nn002 dog\n", encoding="utf-8")
            overrides = root / "prompts.json"
            overrides.write_text('{"n001": "black cat"}', encoding="utf-8")
            self.assertEqual(load_prompts(overrides, synsets),
                             {"n001": "black cat", "n002": "dog"})
            overrides.write_text('{"n001": ""}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_prompts(overrides, synsets)

    def test_only_highest_confidence_instance_is_selected(self):
        masks = torch.tensor([[[[True, False], [False, False]]],
                              [[[False, False], [False, True]]]])
        mask, score, index, count = best_mask(masks, torch.tensor([0.01, 0.09]), 2, 2)
        np.testing.assert_array_equal(mask, [[False, False], [False, True]])
        self.assertAlmostEqual(score, 0.09)
        self.assertEqual((index, count), (1, 2))

    def test_no_detections_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no candidates"):
            best_mask(torch.empty(0, 1, 2, 3), torch.empty(0), 2, 3)

    def test_confidence_ties_select_first_candidate(self):
        masks = torch.tensor([[[1, 0]], [[0, 1]]])
        mask, _, index, _ = best_mask(masks, torch.tensor([0.2, 0.2]), 1, 2)
        self.assertEqual(index, 0)
        np.testing.assert_array_equal(mask, [[True, False]])

    def test_empty_best_mask_does_not_silently_pick_another(self):
        with self.assertRaisesRegex(ValueError, "Highest-confidence SAM mask is empty"):
            best_mask(torch.tensor([[[0]], [[1]]]), torch.tensor([0.9, 0.1]), 1, 1)

    def test_invalid_scores_are_rejected(self):
        for scores in (torch.tensor([float("nan")]), torch.tensor([1.2]), torch.empty(2)):
            with self.assertRaises(ValueError):
                best_mask(torch.ones(1, 1, 1), scores, 1, 1)

    def test_wrong_resolution_is_rejected(self):
        with self.assertRaises(ValueError):
            best_mask(torch.ones(1, 1, 3, 3), torch.ones(1), 2, 2)


class LocalizationTests(unittest.TestCase):
    def test_top_80_percent_has_exact_count_and_stable_ties(self):
        actual = top_k_mask(np.ones((2, 5)), 80)
        np.testing.assert_array_equal(actual.reshape(-1), [True] * 8 + [False] * 2)
        self.assertEqual(int(top_k_mask(np.arange(6).reshape(2, 3), 80).sum()), 5)

    def test_top_k_uses_descending_saliency(self):
        actual = top_k_mask(np.array([[0, 4], [3, 1]]), 50)
        np.testing.assert_array_equal(actual, [[False, True], [True, False]])

    def test_invalid_maps_and_percentages_are_rejected(self):
        for percent in (0, -1, 101, float("nan")):
            with self.assertRaises(ValueError):
                top_k_mask(np.ones((2, 2)), percent)
        with self.assertRaises(ValueError):
            top_k_mask(np.array([[np.inf]]), 80)
        with self.assertRaises(ValueError):
            top_k_mask(np.ones(4), 80)

    def test_iou_values_and_empty_gt(self):
        prediction = np.array([[True, True], [False, False]])
        target = np.array([[True, False], [True, False]])
        self.assertAlmostEqual(binary_iou(prediction, target), 1 / 3)
        self.assertEqual(binary_iou(prediction, prediction), 1)
        self.assertEqual(binary_iou(prediction, ~prediction), 0)
        self.assertEqual(binary_iou(prediction, np.zeros_like(target)), 0)
        with self.assertRaises(ValueError):
            binary_iou(prediction, np.zeros((3, 3), dtype=bool))

    def test_rgb_ramp_is_inverted_not_converted_to_luminance(self):
        values = np.array([[0, 0.1, 0.5, 0.9, 1]])
        rgb = np.stack([np.clip(3 * values - offset, 0, 1)
                        for offset in (0, 1, 2)], axis=-1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray((rgb * 255).astype(np.uint8)).save(root / "clean_map.png")
            actual, source = load_clean_map(root)
            self.assertEqual(source, "inverted_quantized_rgb")
            np.testing.assert_allclose(actual, values, atol=1 / 255)
            torch.save(torch.tensor(values), root / "clean_map.pt")
            actual, source = load_clean_map(root)
            self.assertEqual(source, "raw_tensor")
            np.testing.assert_allclose(actual, values)

    def test_gt_alignment_matches_saved_clean_and_rejects_wrong_image(self):
        from torchvision.models import get_model_weights
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.fromarray(np.full((300, 500, 3), 75, dtype=np.uint8))
            mask = Image.fromarray(np.pad(np.full((200, 200), 255, dtype=np.uint8),
                                         ((50, 50), (150, 150))))
            image.save(root / "source.png")
            mask.save(root / "mask.png")
            weights = get_model_weights("resnet18").DEFAULT.transforms()
            clean = TF.center_crop(TF.resize(image, weights.resize_size), weights.crop_size)
            clean.save(root / "clean.png")
            actual = align_target(root / "source.png", root / "mask.png",
                                  root / "clean.png", "resnet18")
            expected = TF.center_crop(
                TF.resize(mask, weights.resize_size, interpolation=InterpolationMode.NEAREST),
                weights.crop_size,
            )
            np.testing.assert_array_equal(actual, np.asarray(expected) == 255)
            Image.new("RGB", clean.size).save(root / "clean.png")
            with self.assertRaisesRegex(ValueError, "refusing misaligned IoU"):
                align_target(root / "source.png", root / "mask.png",
                             root / "clean.png", "resnet18")


class LocalizationReportingTests(unittest.TestCase):
    def record(self, seed, stem, value, weight=0.0):
        return {
            "context": {
                "seed": seed, "model": "resnet18", "explain_method": "grad_cam",
                "algorithm": "weighted_sum_ga", "w_m": weight, "w_s": 1 - weight,
                "eps": 20, "loss_type": "margin_loss", "strategy": "uniform",
            },
            "identity": ["n001", stem], "protocol": {"top_k_percent": 80},
            "perturbation_metadata": {}, "clean_iou": value,
            "perturbation_iou": value, "clean_rgb_sha256": stem,
            "gt_mask_sha256": "gt", "clean_binary_sha256": stem,
        }

    def test_seed_means_are_equally_weighted_and_clean_is_deduplicated(self):
        records = [self.record("1", "a", 0), self.record("1", "b", 0),
                   self.record("2", "a", 1), self.record("1", "a", 0, weight=0.5)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_reports(records, root, 4)
            folder = next(root.glob("localization_*"))
            report = json.loads((folder / "statistics.json").read_text(encoding="utf-8"))
            clean = next(cell for cell in report["cells"] if cell["approach"] == "clean")
            self.assertEqual(clean["iou"]["mean"], 0.5)
            self.assertEqual(clean["iou"]["n_seeds"], 2)
            self.assertAlmostEqual(clean["iou"]["std"], 2 ** -0.5)
            latex = (folder / "table.tex").read_text(encoding="utf-8")
            self.assertIn(r"80\%", latex)
            self.assertIn(r"grad\_cam", latex)
            self.assertTrue((folder / "samples.csv").is_file())

    def test_duplicate_approach_is_rejected(self):
        record = self.record("1", "a", 0.2)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Duplicate IoU"):
                save_reports([record, record], Path(directory), 4)

    def test_inconsistent_clean_is_rejected(self):
        left = self.record("1", "a", 0.2)
        right = self.record("1", "a", 0.3, weight=0.5)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Inconsistent Clean"):
                save_reports([left, right], Path(directory), 4)

    def test_inconsistent_clean_reports_paths_and_changed_fields(self):
        left = self.record("1", "a", 0.2)
        right = self.record("1", "a", 0.3, weight=0.5)
        left["sample_dir"] = "first_run/class/a"
        right["sample_dir"] = "second_run/class/a"
        right["clean_binary_sha256"] = "different"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as raised:
                save_reports([left, right], Path(directory), 4)
        message = str(raised.exception)
        self.assertIn("first_run/class/a", message)
        self.assertIn("second_run/class/a", message)
        self.assertIn("clean_iou: 0.2 != 0.3", message)
        self.assertIn("clean_binary_sha256:", message)
        self.assertNotIn("gt_mask_sha256:", message)


if __name__ == "__main__":
    unittest.main()
