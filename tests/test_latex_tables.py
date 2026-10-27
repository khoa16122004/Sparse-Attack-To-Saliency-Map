import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace

from evaluate_latex_generate import generate_tables as tables


def attack_run(seed, weight=None, **changes):
    values = dict(
        run_dir=Path(f"seed_{seed}_weight_{weight}"),
        model="resnet50", explain_method="simple_gradient",
        strategy="uniform", eps=20, loss_type="margin_loss", seed=str(seed),
        algorithm="nsgaii" if weight is None else "weighted_sum_ga",
        w_m=weight, w_s=None if weight is None else 1 - weight,
        asr=0.8, spearman=0.2, num_samples_ok=10,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def faith_record(seed, sample, weight, insertion):
    approach = "ours" if weight is None else str(weight)
    score = {
        "fractions": [0.0, 1.0], "ins_curve": [insertion, insertion],
        "del_curve": [0.2, 0.2], "ins_auc": insertion, "del_auc": 0.2,
        "imd": insertion - 0.2,
    }
    return {
        "schema_version": 1, "status": "ok",
        "sample_dir": str(Path(str(seed)) / approach / "class" / sample),
        "context": {
            "model": "resnet50", "explain_method": "simple_gradient",
            "eps": 20, "loss_type": "margin_loss", "seed": str(seed),
            "algorithm": "nsgaii" if weight is None else "weighted_sum_ga",
            "strategy": "uniform", "w_m": weight,
            "w_s": None if weight is None else 1 - weight,
        },
        "protocol": {"curve_values": "raw_softmax_probability", "evaluation_image": "clean"},
        "target_class": 0,
        "maps": {"clean": copy.deepcopy(score), "perturbation": score},
    }


class SeedStatisticsTests(unittest.TestCase):
    def test_attack_h_score_is_calculated_per_seed(self):
        groups = tables.attack_groups([
            attack_run(1, asr=0.2, spearman=0.1),
            attack_run(2, asr=0.9, spearman=0.7),
        ])
        cells = next(iter(groups.values()))
        stat = tables.summarize_cells(cells, tables.METRICS["attack"])[
            ("resnet50", "simple_gradient", "nsgaii")
        ]["h_score"]
        values = [tables.run_score._h_score(0.2, 0.1), tables.run_score._h_score(0.9, 0.7)]
        self.assertAlmostEqual(stat["mean"], mean(values))
        self.assertAlmostEqual(stat["std"], stdev(values))
        self.assertNotAlmostEqual(stat["mean"], tables.run_score._h_score(0.55, 0.4))

    def test_faithfulness_equal_seed_weight_and_clean_deduplication(self):
        records = []
        for weight in (0.0, 0.5, 1.0, None):
            records.extend([
                faith_record(1, "a", weight, 0.2),
                faith_record(2, "a", weight, 0.8),
                faith_record(2, "b", weight, 1.0),
            ])
        cells = next(iter(tables.faithfulness_groups(records).values()))
        self.assertEqual(len(cells), 5)
        stats = tables.summarize_cells(cells, tables.METRICS["faithfulness"])
        stat = stats[("resnet50", "simple_gradient", "clean")]["ins_auc"]
        self.assertAlmostEqual(stat["mean"], 0.55)
        self.assertAlmostEqual(stat["std"], stdev([0.2, 0.9]))
        self.assertEqual(stat["n_seeds"], 2)
        self.assertEqual(len(cells[("resnet50", "simple_gradient", "clean")]["2"]["sample_ids"]), 2)

    def test_configurations_are_not_mixed(self):
        runs = [
            attack_run(1),
            attack_run(2, strategy="saliency_guided"),
            attack_run(3, eps=50),
            attack_run(4, loss_type="negative_cross_entropy_saliency"),
        ]
        self.assertEqual(len(tables.attack_groups(runs)), 4)
        records = [faith_record(1, "a", None, 0.5), faith_record(2, "a", None, 0.5)]
        records[1]["perturbation_metadata"] = {"grid_size": 7}
        self.assertEqual(len(tables.faithfulness_groups(records)), 2)

    def test_duplicate_and_invalid_values_are_rejected(self):
        for runs in (
            [attack_run(1), attack_run(1)],
            [attack_run(1), attack_run(1, run_dir=Path("other"))],
            [attack_run("default")],
            [attack_run(1, spearman=float("nan"))],
        ):
            with self.subTest(runs=runs), self.assertRaises(ValueError):
                tables.attack_groups(runs)

    def test_single_seed_sd_is_undefined(self):
        cells = next(iter(tables.attack_groups([attack_run(1)]).values()))
        latex, report = tables.render_table("attack", cells)
        self.assertIn(r"0.8000 \pm --", latex)
        self.assertIsNone(report["cells"][0]["metrics"]["asr"]["std"])
        self.assertNotIn(r"\textcolor{blue}{\mathbf", latex)


class PairedTests(unittest.TestCase):
    def cells(self, last_baseline_better=False):
        runs = []
        for seed in range(1, 6):
            for weight in (0.0, 0.5, 1.0, None):
                value = 0.9 if weight is None else 0.4
                if last_baseline_better and weight == 1.0:
                    value = 0.99
                runs.append(attack_run(seed, weight, asr=value, spearman=1 - value))
        return next(iter(tables.attack_groups(runs).values()))

    def test_significance_requires_all_three_and_improvement(self):
        latex, report = tables.render_table("attack", self.cells())
        self.assertIn(r"\textcolor{blue}{\mathbf{0.9000", latex)
        tests = [test for test in report["ttest"]["comparisons"] if test["n_pairs"] == 5]
        self.assertEqual(len(tests), 9)
        self.assertTrue(all(test["p_value"] == 0 for test in tests))
        latex, _ = tables.render_table("attack", self.cells(last_baseline_better=True))
        self.assertNotIn(r"\textcolor{blue}{\mathbf", latex)

    def test_pairing_uses_seed_identity_not_insertion_order(self):
        from scipy.stats import ttest_rel
        ours = {
            str(seed): {"metrics": {"asr": value}, "sample_ids": None}
            for seed, value in [(1, 0.5), (2, 0.7), (3, 0.9)]
        }
        baseline = {
            str(seed): {"metrics": {"asr": value}, "sample_ids": None}
            for seed, value in [(3, 0.4), (1, 0.3), (2, 0.6)]
        }
        result = tables.paired_comparison(ours, baseline, "asr", True, ttest_rel)
        expected = ttest_rel([0.5, 0.7, 0.9], [0.3, 0.6, 0.4]).pvalue
        self.assertAlmostEqual(result["p_value"], expected)
        baseline["2"]["sample_ids"] = [["class", "other"]]
        result = tables.paired_comparison(ours, baseline, "asr", True, ttest_rel)
        self.assertEqual(result["status"], "unmatched_samples")
        self.assertIsNone(result["p_value"])

    def test_disabled_tests_remove_color_and_caption_claim(self):
        latex, report = tables.render_table("attack", self.cells(), use_ttest=False)
        self.assertNotIn(r"\textcolor{blue}", latex)
        self.assertNotIn("t-tests", latex)
        self.assertEqual(report["ttest"]["comparisons"], [])
        self.assertIn(r"\mathbf{0.9000", latex)

    def test_bonferroni_threshold(self):
        cells = self.cells()
        for seed, value in zip(range(1, 6), [0.01, 0.02, 0.03, 0.04, 0.13]):
            cells[("resnet50", "simple_gradient", "nsgaii")][str(seed)]["metrics"]["asr"] = 0.4 + value
        _, report = tables.render_table("attack", cells, correction="bonferroni")
        tests = [
            test for test in report["ttest"]["comparisons"]
            if test["n_pairs"] == 5 and test["metric"] == "asr"
        ]
        from scipy.stats import ttest_rel
        expected = float(ttest_rel([0.01, 0.02, 0.03, 0.04, 0.13], [0] * 5).pvalue)
        self.assertTrue(all(math.isclose(test["p_value"], expected) for test in tests))
        latex, _ = tables.render_table("attack", cells, correction="bonferroni")
        self.assertIn("Bonferroni corrected", latex)


class TemplateTests(unittest.TestCase):
    def test_structure_and_fixed_models(self):
        for kind in ("attack", "faithfulness"):
            with self.subTest(kind=kind):
                latex, report = tables.render_table(kind, {}, use_ttest=False)
                self.assertEqual(latex.count(r"\begin{tabular}"), 2)
                expected_rows = 32 if kind == "attack" else 40
                rows = [line for line in latex.splitlines() if line.startswith("& ") and "--" in line]
                self.assertEqual(len(rows), expected_rows)
                self.assertTrue(all(line.count("&") == 10 for line in rows))
                self.assertEqual(len(report["missing_cells"]), expected_rows * 3)
                self.assertIn("Densenet-121" if kind == "attack" else "EfficientNet-B0", latex)
                self.assertNotIn("five independent random seeds", latex)

    def test_cli_writes_both_outputs_and_does_not_change_template(self):
        template = tables.TEMPLATE_DIR / "faithfulness.text"
        original = template.read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed in (1, 2):
                path = root / str(seed) / "faithfulness.json"
                path.parent.mkdir()
                path.write_text(json.dumps(faith_record(seed, "a", None, 0.5)))
            tables.main([
                "--faithfulness-input-dirs", str(root),
                "--output-dir", str(root / "output"), "--no-ttest",
            ])
            outputs = list((root / "output").iterdir())
            self.assertEqual(len(outputs), 1)
            self.assertTrue((outputs[0] / "table.tex").is_file())
            report = json.loads((outputs[0] / "statistics.json").read_text())
            self.assertEqual(report["configuration"]["eps"], 20)
            self.assertEqual(report["cells"][0]["metrics"]["imd"]["n_seeds"], 2)
        self.assertEqual(template.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
