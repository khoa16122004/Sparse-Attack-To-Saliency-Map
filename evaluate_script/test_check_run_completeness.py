import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluate_script.check_run_completeness import (
    Combination, InventoryRun, build_report, expected_grid, load_cache, scan_root,
)


class CompletenessTests(unittest.TestCase):
    def test_standard_grid_and_missing_entire_seed(self):
        args = argparse.Namespace(
            seed="all", model="all", explain_method="all", algorithm="all", eps="all",
            expected_seeds=[str(seed) for seed in range(22520691, 22520696)],
            expected_models=["resnet18", "vgg16", "vit_b_32"],
            expected_explain_methods=[
                "simple_gradient", "input_gradient", "grad_cam", "raw_attention"
            ],
        )
        grid = expected_grid(args)
        self.assertEqual(len(grid), 1440)
        report = build_report([], grid)
        self.assertEqual(report["summary"], {"missing": 1440})
        self.assertEqual(len(report["groups"]), 60)

    def test_threshold_duplicates_and_equivalent_loss(self):
        base = Combination(
            "22520692", "resnet18", "grad_cam", "weighted_sum_ga",
            0.0, 1.0, 20, "margin_loss",
        )
        other = Combination(**{
            **base.__dict__, "loss_type": "negative_cross_entropy_saliency"
        })
        for samples, status in [
            (0, "insufficient"), (49, "insufficient"), (50, "partial"),
            (99, "partial"), (100, "complete"),
        ]:
            with self.subTest(samples=samples):
                report = build_report([
                    InventoryRun(base, "old", 1, 0, []),
                    InventoryRun(base, "new", samples, samples, []),
                ], [base, other])
                self.assertEqual(report["combinations"][0]["status"], status)
                self.assertEqual(report["combinations"][1]["status"], "missing")
                self.assertEqual(
                    report["combinations"][1]["equivalent_loss_available"], "margin_loss"
                )

    def test_scan_empty_corrupt_and_duplicate_summary_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "GA" / "22520692" / "resnet18"
            name = (
                "strategy-uniform__wm-0.5__ws-0.5__eps-20__"
                "exp-grad_cam__seed-22520692"
            )
            run = model / name
            sample = run / "class" / "sample"
            sample.mkdir(parents=True)
            (sample / "summary.json").write_text('{"status":"ok"}', encoding="utf-8")
            (sample / "summarize.json").write_text('{"status":"ok"}', encoding="utf-8")
            broken = run / "class" / "broken"
            broken.mkdir()
            (broken / "summary.json").write_text("{", encoding="utf-8")
            (model / name.replace("eps-20", "eps-50")).mkdir()
            runs, issues = scan_root(root, check_samples=True)
            self.assertEqual(issues, [])
            self.assertEqual(len(runs), 2)
            populated = next(item for item in runs if item.samples_total)
            self.assertEqual(populated.samples_total, 2)
            self.assertEqual(populated.samples_ok, 1)
            self.assertEqual(len(populated.errors), 1)
            self.assertEqual(populated.combination.seed, "22520692")
            self.assertEqual(populated.combination.algorithm, "weighted_sum_ga")
            self.assertEqual(populated.combination.model, "resnet18")

    def test_default_scan_does_not_read_or_visit_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "GA" / "22520692" / "resnet18" / (
                "strategy-uniform__wm-0.5__ws-0.5__eps-20__"
                "exp-grad_cam__seed-22520692"
            )
            run.mkdir(parents=True)
            with patch.object(Path, "glob", side_effect=AssertionError("Visited samples")):
                with patch.object(Path, "open", side_effect=AssertionError("Read a file")):
                    runs, issues = scan_root(root)
            self.assertEqual(issues, [])
            self.assertEqual(len(runs), 1)
            report = build_report(runs, [runs[0].combination])
            row = report["combinations"][0]
            self.assertEqual(row["status"], "present")
            self.assertIsNone(row["samples_ok"])
            self.assertIsNone(row["usable_for_score"])

    def test_cache_preserves_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "all_runs.json"
            cache.write_text(json.dumps([{
                "seed": "22520692", "model": "resnet18",
                "explain_method": "grad_cam", "algorithm": "ga",
                "strategy": "uniform", "w_m": 0.5, "w_s": 0.5,
                "eps": 20, "loss_type": "margin_loss", "run_dir": "not-on-disk",
                "num_samples_ok": 0, "num_samples_total": 10,
            }]), encoding="utf-8")
            runs = load_cache(cache)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].samples_ok, 0)


if __name__ == "__main__":
    unittest.main()
