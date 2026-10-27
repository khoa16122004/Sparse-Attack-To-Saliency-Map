import copy
import json
import tempfile
import unittest
from pathlib import Path
from statistics import mean, stdev

from evaluate_latex_generate import plot_epsilon_curves as curves


def report(eps):
    cells = []
    for approach in curves.APPROACHES:
        for model, offset in (("resnet50", 0.0), ("vgg16", 0.2)):
            cells.append({
                "model": model,
                "explain_method": "simple_gradient",
                "approach": approach,
                "metrics": {
                    metric: {
                        # Deliberately wrong summaries: aggregation must use seed values.
                        "mean": 99,
                        "std": 99,
                        "seed_values": {
                            str(seed): offset + seed * 0.01 * (1 if offset == 0 else -1)
                            for seed in range(1, 6)
                        },
                    }
                    for metric in curves.METRICS
                },
            })
    return {
        "configuration": {"eps": eps, "strategy": "uniform", "loss_type": "margin_loss"},
        "averaging": "equal_weight_seed_means",
        "cells": cells,
    }


class EpsilonCurveTests(unittest.TestCase):
    def aggregate(self, reports):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, value in enumerate(reports):
                path = Path(directory) / f"{index}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                paths.append(path)
            return curves.aggregate_reports(paths)

    def test_seed_averaging_precedes_sd_and_epsilon_is_sorted(self):
        summary = self.aggregate([report(100), report(20), report(50)])
        self.assertEqual([point["eps"] for point in summary["points"]], [20, 50, 100])
        for metric in curves.METRICS:
            stats = summary["points"][0]["approaches"]["ga_0"][metric]
            self.assertAlmostEqual(stats["mean"], 0.1)
            self.assertAlmostEqual(stats["std"], 0)
            self.assertEqual(stats["n_seeds"], 5)
        self.assertEqual(len(summary["coverage"]), 2)

    def test_sample_sd_of_seed_means(self):
        first = report(20)
        for cell in first["cells"]:
            for metric in curves.METRICS:
                cell["metrics"][metric]["seed_values"] = {
                    str(seed): seed / 10 for seed in range(1, 6)
                }
        stats = self.aggregate([first, report(50)])["points"][0]["approaches"]["nsgaii"]["asr"]
        self.assertAlmostEqual(stats["mean"], mean([0.1, 0.2, 0.3, 0.4, 0.5]))
        self.assertAlmostEqual(stats["std"], stdev([0.1, 0.2, 0.3, 0.4, 0.5]))

    def test_plot_has_three_panels_and_four_shaded_curves(self):
        summary = self.aggregate([report(100), report(20), report(50)])
        figure = curves.plot_curves(summary)
        import matplotlib.pyplot as plt

        try:
            self.assertEqual(len(figure.axes), 3)
            for axis in figure.axes:
                self.assertEqual(len(axis.lines), 4)
                self.assertEqual(len(axis.collections), 4)
                self.assertEqual(list(axis.get_xticks()), [0, 1, 2])
                self.assertEqual(
                    [tick.get_text() for tick in axis.get_xticklabels()], ["20", "50", "100"]
                )
                self.assertEqual(axis.get_xlabel(), r"$\epsilon$")
                self.assertEqual(axis.get_title(), "")
                for approach, line in zip(curves.APPROACHES, axis.lines):
                    self.assertEqual(list(line.get_xdata()), [0, 1, 2])
                    self.assertEqual(line.get_marker(), "o")
                    self.assertEqual(line.get_markerfacecolor(), "white")
                    self.assertEqual(line.get_color(), curves.PLOT_COLORS[approach])
                    self.assertEqual(line.get_markeredgecolor(), curves.PLOT_COLORS[approach])
                self.assertEqual(axis.lines[-1].get_color(), "#9B1C31")
                self.assertGreater(axis.lines[-1].get_linewidth(), axis.lines[0].get_linewidth())
                self.assertTrue(all(spine.get_visible() for spine in axis.spines.values()))
            self.assertEqual(len(figure.legends[0].get_texts()), 4)
            self.assertEqual(len(figure.texts), 0)
            self.assertEqual(list(figure.get_size_inches()), [9, 3.2])
        finally:
            plt.close(figure)

    def test_invalid_or_unbalanced_reports_are_rejected(self):
        invalid = []
        changed = report(50)
        changed["cells"].pop()
        invalid.append(changed)
        changed = report(50)
        changed["configuration"]["strategy"] = "saliency_guided"
        invalid.append(changed)
        for value in (float("nan"), None, True):
            changed = report(50)
            changed["cells"][0]["metrics"]["asr"]["seed_values"]["1"] = value
            invalid.append(changed)
        changed = report(50)
        del changed["cells"][0]["metrics"]["asr"]["seed_values"]["1"]
        invalid.append(changed)
        changed = report(50)
        seeds = changed["cells"][0]["metrics"]["asr"]["seed_values"]
        seeds["other"] = seeds.pop("1")
        invalid.append(changed)
        changed = report(50)
        changed["cells"].append(copy.deepcopy(changed["cells"][0]))
        invalid.append(changed)
        invalid.append(report(20))
        for changed in invalid:
            with self.subTest(report=changed), self.assertRaises(ValueError):
                self.aggregate([report(20), changed])
        with self.assertRaises(ValueError):
            self.aggregate([])


if __name__ == "__main__":
    unittest.main()
