import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from evaluate_script.paretofront_compare_ui import (
    SampleFronts, discover_runs, export_paper_figure, load_samples,
    make_figure, make_paper_figure, rank_comparisons,
)


def write_front(run: Path, sample: str, content: str) -> Path:
    path = run / "class" / sample / "non_dominated_front_scores.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class ExplorerTests(unittest.TestCase):
    def setUp(self):
        discover_runs.clear()
        load_samples.clear()

    def test_pairwise_missing_and_bad_files_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ns, ga, other = root / "ns", root / "ga", root / "other"
            write_front(ns, "shared", "0 0\n")
            write_front(ns, "ns_only", "0 0\n")
            write_front(ns, "bad", "0 0\n")
            write_front(ga, "shared", "1 1\n1 1\n")
            write_front(ga, "ga_only", "1 1\n")
            write_front(ga, "bad", "1 1\ncorrupt\n")
            other.mkdir()
            samples, issues = load_samples([
                (20, "nsgaii", str(ns)), (20, "ga_0.5_0.5", str(ga)),
                (20, "ga_1.0_0.0", str(other)),
            ])
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].sample_id, "class/shared")
            self.assertNotIn("ga_1.0_0.0", samples[0].fronts)
            self.assertEqual(len(samples[0].fronts["ga_0.5_0.5"]), 1)
            self.assertTrue(any("only one side" in issue for issue in issues))
            self.assertTrue(any("corrupt" not in issue and ":2:" in issue for issue in issues))
            self.assertEqual(rank_comparisons(samples)[0].score.coverage, 1)

    def test_ranking_net_advantage_before_forward_coverage(self):
        samples = [
            SampleFronts(20, "tradeoff", {
                "nsgaii": [(0, 2), (3, 0)], "ga_0.5_0.5": [(1, 3), (2, 0)],
            }, {}),
            SampleFronts(50, "winner", {
                "nsgaii": [(0, 0)], "ga_0.5_0.5": [(1, 1)],
            }, {}),
        ]
        ranked = rank_comparisons(samples)
        self.assertEqual([r.sample.sample_id for r in ranked], ["winner", "tradeoff"])
        self.assertEqual(ranked[1].score.advantage, 0)
        self.assertEqual(ranked[0].row()["Net advantage (pp)"], 100)

    def test_scatter_hollow_compact_zoom_and_scaling(self):
        sample = SampleFronts(20, "sample", {
            "nsgaii": [(-10, 0.2), (-9, 0.1)],
            "ga_0.5_0.5": [(-8, 0.3)],
        }, {})
        figure = make_figure(sample, list(sample.fronts), "margin_loss", 8, 420, False)
        self.assertEqual(figure.layout.height, 420)
        self.assertEqual(figure.layout.dragmode, "zoom")
        self.assertEqual(figure.data[0].marker.symbol, "circle")
        self.assertEqual(figure.data[0].marker.color, "white")
        self.assertEqual(figure.data[0].marker.line.color, figure.data[0].line.color)
        self.assertEqual(figure.data[-1].mode, "lines+markers")
        self.assertEqual(list(figure.data[-1].x), [-10, -9])
        self.assertAlmostEqual(figure.layout.xaxis.range[0], -10.08)
        scaled = make_figure(sample, list(sample.fronts), "negative_cross_entropy_saliency", 8, 420, True)
        self.assertEqual(list(scaled.data[-1].x), [0, 0.5])
        self.assertIn(r"\mathcal{L}_{\text{attack}}", scaled.layout.xaxis.title.text)
        self.assertEqual(tuple(scaled.data[-1].customdata[0]), (-10, 0.2))
        flat = SampleFronts(20, "flat", {"nsgaii": [(0, 0)]}, {})
        flat_figure = make_figure(flat, ["nsgaii"], "margin_loss", 8, 420, True)
        self.assertLess(*flat_figure.layout.xaxis.range)
        with self.assertRaises(ValueError):
            make_figure(sample, [], "margin_loss", 8, 420, False)

    def test_discovery_keeps_duplicate_runs_and_loss_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = "strategy-uniform__eps-20__exp-grad_cam__wm-0.5__ws-0.5"
            for suffix in ["__repeat-1", "__repeat-2", "__fit-negative_cross_entropy_saliency"]:
                (root / "22520691" / "resnet18" / (prefix + suffix)).mkdir(parents=True)
            runs = discover_runs(str(root), "ga")
            self.assertEqual(len(runs), 3)
            self.assertEqual(sum(r.loss_type == "margin_loss" for r in runs), 2)
            with self.assertRaises(ValueError):
                discover_runs(str(root / "missing"), "ga")

    def test_paper_figure_and_exports(self):
        sample = SampleFronts(20, "sample", {
            "nsgaii": [(-10, 0.2), (-9, 0.1)],
            "ga_0.5_0.5": [(-8, 0.3)],
        }, {})
        figure = make_paper_figure(sample, list(sample.fronts), "margin_loss", 8, 760, 420, False)
        ax = figure.axes[0]
        self.assertEqual(ax.get_title(), "")
        self.assertEqual(ax.get_xlabel(), r"$\mathcal{L}_{\text{attack}}$")
        self.assertEqual(ax.get_ylabel(), r"$\mathcal{L}_{\text{explain}}$")
        self.assertEqual(len(ax.lines), 2)
        self.assertEqual(ax.lines[0].get_markerfacecolor(), "white")
        self.assertEqual(ax.get_legend().texts[-1].get_text(), r"$\mathbf{Ours}$")
        self.assertEqual(ax.get_legend().texts[0].get_text(), r"GASS ($\lambda=0.5$)")
        self.assertEqual(ax.lines[-1].get_color(), "#9B1C31")
        self.assertAlmostEqual(ax.get_xlim()[0], -10.08)
        png = export_paper_figure(figure, "png")
        self.assertTrue(png.startswith(b"\x89PNG"))
        with Image.open(io.BytesIO(png)) as image:
            self.assertEqual(image.size, (3800, 2100))
            self.assertAlmostEqual(image.info["dpi"][0], 600, places=1)
        self.assertTrue(export_paper_figure(figure, "pdf").startswith(b"%PDF"))
        self.assertIn(b"<svg", export_paper_figure(figure, "svg"))

    def test_streamlit_navigation_filters_and_chart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = "strategy-uniform__eps-20__exp-grad_cam__wm-0.5__ws-0.5"
            ga = root / "GA" / "22520691" / "resnet18" / prefix
            ns = root / "NSGAII" / "22520691" / "resnet18" / (prefix + "__algo-nsgaii")
            for sample, ns_score in [("best", "0 0\n"), ("tradeoff", "0 2\n")]:
                write_front(ga, sample, "1 1\n")
                write_front(ns, sample, ns_score)
            run_dir = Path(__file__).with_name("paretofront_compare_ui.py")
            # AppTest uses an importable package wrapper, matching the Streamlit entry point.
            app = AppTest.from_string(
                "from evaluate_script.paretofront_compare_ui import main\nmain()",
                default_timeout=30,
            )
            with patch(
                "evaluate_script.paretofront_compare_ui.discover_runs",
                side_effect=lambda _root, algorithm: discover_runs(
                    str(root / ("GA" if algorithm == "ga" else "NSGAII")), algorithm,
                ),
            ):
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(run_dir.is_file())
                self.assertEqual(app.metric[0].value, "100.0%")
                self.assertEqual(len(app.get("imgs")), 1)
                self.assertEqual(
                    next(s for s in app.slider if s.label == "Square figure size (px)").value,
                    380,
                )
                self.assertTrue(any(b.label == "Download PDF" for b in app.get("download_button")))
                next(r for r in app.radio if r.label == "View").set_value("Interactive").run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get("plotly_chart")), 1)
                next(button for button in app.button if button.label == "Next").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.metric[0].value, "0.0%")
                next(button for button in app.button if button.label == "Previous").click().run()
                self.assertEqual(app.metric[0].value, "100.0%")
                coverage = next(s for s in app.slider if s.label == "Minimum GA points dominated (%)")
                coverage.set_value(80).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.selectbox(key="comparison").options), 1)
                search = next(t for t in app.text_input if t.label == "Search class / sample")
                search.set_value("missing").run()
                self.assertFalse(app.exception)
                self.assertTrue(any("No pairs" in info.value for info in app.info))

    def test_upper_right_legend_fits_and_fronts_are_sorted(self):
        sample = SampleFronts(20, "sample", {
            "nsgaii": [(2, 0), (0, 2), (1, 1)],
            "ga_1.0_0.0": [(3, 2)],
            "ga_0.0_1.0": [(2, 3)],
            "ga_0.5_0.5": [(3, 3)],
        }, {})
        for width in (360, 420, 760):
            figure = make_paper_figure(
                sample, list(sample.fronts), "margin_loss", 8, width, width, False,
            )
            self.assertEqual(figure.get_figwidth(), figure.get_figheight())
            with Image.open(io.BytesIO(export_paper_figure(figure, "png"))) as image:
                self.assertEqual(image.width, image.height)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            legend = figure.axes[0].get_legend()
            bounds = legend.get_window_extent(renderer)
            ax = figure.axes[0]
            axes_bounds = ax.get_window_extent(renderer)
            self.assertAlmostEqual(axes_bounds.width, axes_bounds.height)
            self.assertGreaterEqual(bounds.x0, axes_bounds.x0)
            self.assertLess(bounds.x1, axes_bounds.x1)
            self.assertLess(bounds.y1, axes_bounds.y1)
            self.assertGreater(bounds.x1, axes_bounds.x0 + axes_bounds.width * 0.9)
            self.assertGreater(bounds.y1, axes_bounds.y0 + axes_bounds.height * 0.9)
            self.assertEqual(legend.texts[0].get_fontsize(), 10)
            for ticks, limits in (
                (ax.get_xticks(), ax.get_xlim()), (ax.get_yticks(), ax.get_ylim()),
            ):
                visible_ticks = [v for v in ticks if limits[0] <= v <= limits[1]]
                self.assertLessEqual(len(visible_ticks), 4)
                self.assertGreaterEqual(len(visible_ticks), 2)
            self.assertEqual(ax.xaxis.label.get_fontsize(), 16)
            self.assertEqual(ax.yaxis.label.get_fontsize(), 16)
            self.assertGreaterEqual(bounds.x0, 0)
            self.assertLessEqual(bounds.x1, figure.bbox.width)
            text_y = [text.get_window_extent(renderer).y0 for text in legend.texts]
            self.assertTrue(all(a > b for a, b in zip(text_y, text_y[1:])))
            self.assertEqual(list(figure.axes[0].lines[-1].get_xdata()), [0, 1, 2])
            self.assertEqual(
                [text.get_text() for text in legend.texts],
                [r"GASS ($\lambda=0$)", r"GASS ($\lambda=0.5$)",
                 r"GASS ($\lambda=1$)", r"$\mathbf{Ours}$"],
            )

    def test_round_ticks_on_paper_example(self):
        sample = SampleFronts(20, "example", {
            "nsgaii": [(-7.6, 0.77), (-4, 0.58)],
            "ga_0.0_1.0": [(-1.7, 0.66), (-1.3, 0.58)],
            "ga_0.5_0.5": [(-7.4, 0.86), (-6.7, 0.84)],
            "ga_1.0_0.0": [(-7.2, 0.8), (-5.7, 0.77)],
        }, {})
        figure = make_paper_figure(sample, list(sample.fronts), "margin_loss", 8, 420, 420, False)
        ax = figure.axes[0]
        for ticks, limits, expected in (
            (ax.get_xticks(), ax.get_xlim(), [-6, -4, -2]),
            (ax.get_yticks(), ax.get_ylim(), [0.6, 0.7, 0.8]),
        ):
            visible = [float(v) for v in ticks if limits[0] <= v <= limits[1]]
            self.assertEqual(len(visible), len(expected))
            for value, target in zip(visible, expected):
                self.assertAlmostEqual(value, target)


if __name__ == "__main__":
    unittest.main()
