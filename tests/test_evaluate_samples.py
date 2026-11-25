import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import evaluate
from evaluate_script.generate_imagenet_masks import sha256


class EvaluateSamplesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sample = (self.root / "runs" / "GA" / "1" / "resnet18"
                       / "strategy-uniform__wm-0.5__ws-0.5__eps-20__exp-simple_gradient"
                       / "n001" / "image")
        self.sample.mkdir(parents=True)
        self.summary_path = self.sample / "summary.json"
        self.summary = {
            "status": "ok", "true_label": 0, "clean_pred": 0, "adv_pred": 1,
            "first_success_iteration": 2, "history_margin": [1.0, 0.5],
            "history_saliency": [0.8, 0.2],
        }
        self.summary_path.write_text(json.dumps(self.summary), encoding="utf-8")
        clean = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        Image.fromarray(clean).save(self.sample / "clean_map.png")
        Image.fromarray(255 - clean).save(self.sample / "adv_map.png")
        Image.fromarray(np.repeat(clean[:, :, None], 3, axis=2)).save(self.sample / "clean.png")
        torch.save(torch.tensor([[0., 1.], [2., 3.]]), self.sample / "clean_map.pt")
        torch.save({"smoothed_density": torch.tensor([[3., 2.], [1., 0.]])},
                   self.sample / "perturbation_density.pt")

        self.images = self.root / "images"
        self.masks = self.root / "masks"
        (self.images / "n001").mkdir(parents=True)
        (self.masks / "n001").mkdir(parents=True)
        source = self.images / "n001" / "image.png"
        mask = self.masks / "n001" / "image.png"
        Image.fromarray(clean).save(source)
        Image.fromarray(np.array([[0, 0], [255, 255]], dtype=np.uint8)).save(mask)
        manifest = {"schema_version": 1, "status": "complete", "images": {
            "n001/image.png": {"mask": "n001/image.png", "source_sha256": sha256(source),
                               "mask_sha256": sha256(mask)},
        }}
        (self.masks / "mask_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def arguments(self):
        return [
            "--input-dirs", str(self.root / "runs"), "--imagenet-root", str(self.images),
            "--mask-root", str(self.masks), "--device", "cpu", "--step", "2",
            "--batch-size", "2",
        ]

    def test_end_to_end_sample_metrics_without_aggregation(self):
        import util

        class ToyModel(torch.nn.Module):
            def forward(self, image):
                score = image.mean(dim=(1, 2, 3))
                return torch.stack((score, -score), dim=1)

        target = np.array([[False, False], [True, True]])
        explain = lambda model, image, normalize, target: (
            torch.tensor([[[0., 1.], [2., 3.]]]), None,
        )
        with (
            patch.object(evaluate, "align_target", return_value=target),
            patch.object(util, "get_torchvision_model",
                         return_value=(ToyModel(), None, lambda x: x)) as get_model,
            patch.object(util, "get_explainable_method", return_value=explain),
        ):
            evaluate.main(self.arguments())
            output = json.loads((self.sample / "evaluation.json").read_text(encoding="utf-8"))
            self.assertTrue(output["attack"]["success"])
            self.assertAlmostEqual(output["attack"]["spearman"], -1.0)
            self.assertEqual(output["attack"]["first_success_iteration"], 2)
            self.assertEqual(output["context"]["w_m"], 0.5)
            self.assertEqual(output["localization"]["clean_iou"], 1.0)
            self.assertEqual(output["localization"]["perturbation_iou"], 0.0)
            for metrics in output["faithfulness"]["maps"].values():
                self.assertEqual(metrics["fractions"], [0.0, 0.5, 1.0])
                self.assertAlmostEqual(metrics["imd"], metrics["ins_auc"] - metrics["del_auc"])
            self.assertFalse((self.root / "faithfulness_aggregates").exists())
            output_path = self.sample / "evaluation.json"
            original = output_path.read_bytes()
            evaluate.main(self.arguments())
            self.assertEqual(get_model.call_count, 1)
            self.assertEqual(output_path.read_bytes(), original)

            second = self.sample.parent / "image2"
            shutil.copytree(self.sample, second, ignore=shutil.ignore_patterns("evaluation.json"))
            for root in (self.images, self.masks):
                shutil.copyfile(root / "n001" / "image.png", root / "n001" / "image2.png")
            manifest_path = self.masks / "mask_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            item = dict(manifest["images"]["n001/image.png"])
            item["mask"] = "n001/image2.png"
            manifest["images"]["n001/image2.png"] = item
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            evaluate.main(self.arguments())
            self.assertEqual(get_model.call_count, 2)
            self.assertEqual(output_path.read_bytes(), original)
            self.assertTrue((second / "evaluation.json").is_file())
            with self.assertRaisesRegex(ValueError, "settings changed.*--replace"):
                evaluate.main(self.arguments() + ["--threshold", "0.8"])
            output_path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid or incompatible"):
                evaluate.main(self.arguments())
            evaluate.main(self.arguments() + ["--replace"])
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["status"], "ok")

    def test_undefined_spearman_is_explicit_and_class_fallback_is_supported(self):
        Image.new("L", (2, 2), 0).save(self.sample / "clean_map.png")
        metrics = evaluate.attack_metrics(self.summary_path, {"clean_pred": 3, "adv_pred": 3})
        self.assertFalse(metrics["success"])
        self.assertIsNone(metrics["spearman"])
        self.assertEqual(metrics["spearman_error"], "constant_or_too_small_map")
        with self.assertRaisesRegex(ValueError, "adv_pred"):
            evaluate.attack_metrics(self.summary_path, {"true_label": 1})

    def test_changed_gt_and_mismatched_map_shape_are_rejected(self):
        index, digest = evaluate.load_mask_index(self.masks)
        args = SimpleNamespace(imagenet_root=self.images, mask_root=self.masks, empty_gt="error",
                               threshold=0.5, top_k_percent=None)
        context = {"model": "resnet18"}
        (self.masks / "n001" / "image.png").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "changed"):
            evaluate.localization_metrics(self.summary_path, context, args, index, digest)
        index[("n001", "image")][1]["mask_sha256"] = sha256(self.masks / "n001" / "image.png")
        with patch.object(evaluate, "align_target", return_value=np.ones((1, 1), dtype=bool)):
            with self.assertRaisesRegex(ValueError, "shapes differ"):
                evaluate.localization_metrics(self.summary_path, context, args, index, digest)


if __name__ == "__main__":
    unittest.main()
