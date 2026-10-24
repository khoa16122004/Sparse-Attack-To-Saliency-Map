import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from evaluate_script import run_evaluate_faithfulness as evaluator
from evaluate_script.run_aggregate_faithfulness import configuration, summarize


class SharedCleanTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            device="cpu", step=1, batch_size=2, kernel_size=11, kernel_sigma=5,
        )
        self.context = {
            "model": "resnet18", "explain_method": "simple_gradient", "eps": 20,
            "loss_type": "margin_loss", "algorithm": "weighted_sum_ga",
            "strategy": "uniform", "w_m": 0.0, "w_s": 1.0, "seed": "1",
        }
        self.score = {
            "fractions": [0.0, 1.0], "ins_curve": [0.1, 0.9],
            "del_curve": [0.9, 0.1], "ins_auc": 0.5, "del_auc": 0.5, "imd": 0.0,
        }
        self.image = torch.zeros(1, 3, 2, 2)
        self.explain = Mock(return_value=(torch.ones(1, 2, 2), None))
        self.cache = {}

    def evaluate(self, context, name, label=0):
        return evaluator.evaluate_sample(
            Path(name) / "class" / "sample" / "summary.json",
            {"true_label": label}, context, Mock(), Mock(), self.explain, Mock(),
            self.args, clean_cache=self.cache,
        )

    def test_four_approaches_share_one_clean_evaluation(self):
        with (
            patch.object(evaluator, "load_image", return_value=self.image),
            patch.object(evaluator.torch, "load", return_value={
                "smoothed_density": torch.ones(2, 2),
            }),
            patch.object(evaluator, "evaluate_map", side_effect=lambda *a, **kw: copy.deepcopy(self.score)) as metric,
        ):
            records = []
            for index, weight in enumerate((0.0, 0.5, 1.0, None)):
                context = dict(self.context)
                if weight is None:
                    context["algorithm"] = "nsgaii"
                else:
                    context.update(w_m=weight, w_s=1 - weight)
                records.append(self.evaluate(context, f"approach_{index}"))
        self.assertEqual(self.explain.call_count, 1)
        self.assertEqual(metric.call_count, 5)  # One Clean and four density evaluations.
        self.assertTrue(all(record["maps"]["clean"] is records[0]["maps"]["clean"]
                            for record in records))
        rows = summarize(records)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["n"] == 1 for row in rows))

    def test_different_clean_image_is_rejected(self):
        with (
            patch.object(evaluator, "load_image", side_effect=[self.image, self.image + 0.1]),
            patch.object(evaluator.torch, "load", return_value={
                "smoothed_density": torch.ones(2, 2),
            }),
            patch.object(evaluator, "evaluate_map", return_value=self.score),
        ):
            self.evaluate(self.context, "ga")
            with self.assertRaisesRegex(ValueError, "clean image or true_label differs"):
                self.evaluate(self.context, "nsga")

    def test_different_target_is_rejected(self):
        with (
            patch.object(evaluator, "load_image", return_value=self.image),
            patch.object(evaluator.torch, "load", return_value={
                "smoothed_density": torch.ones(2, 2),
            }),
            patch.object(evaluator, "evaluate_map", return_value=self.score),
        ):
            self.evaluate(self.context, "ga")
            with self.assertRaisesRegex(ValueError, "clean image or true_label differs"):
                self.evaluate(self.context, "nsga", label=1)

    def test_legacy_gamma_is_not_used_for_evaluation_or_grouping(self):
        smoothed = torch.tensor([[0.0, 0.25], [0.5, 1.0]])
        with (
            patch.object(evaluator, "load_image", return_value=self.image),
            patch.object(evaluator.torch, "load", return_value={
                "smoothed_density": smoothed,
                "normalized_density": smoothed.square(),
            }),
            patch.object(evaluator, "evaluate_map", return_value=self.score) as metric,
        ):
            result = self.evaluate(self.context, "ga")
        torch.testing.assert_close(metric.call_args.args[3], smoothed)
        self.assertEqual(result["protocol"]["perturbation_source"], "minmax_smoothed_density_tensor")
        legacy = copy.deepcopy(result)
        legacy["perturbation_metadata"]["gamma"] = 2.0
        self.assertEqual(configuration(legacy), configuration(result))
        legacy["perturbation_metadata"]["gamma"] = None
        self.assertEqual(configuration(legacy), configuration(result))

    def test_missing_smoothed_density_is_reported(self):
        with (
            patch.object(evaluator, "load_image", return_value=self.image),
            patch.object(evaluator.torch, "load", return_value={
                "normalized_density": torch.ones(2, 2),
            }),
            patch.object(evaluator, "evaluate_map", return_value=self.score),
        ):
            with self.assertRaisesRegex(ValueError, "missing smoothed_density"):
                self.evaluate(self.context, "ga")


if __name__ == "__main__":
    unittest.main()
