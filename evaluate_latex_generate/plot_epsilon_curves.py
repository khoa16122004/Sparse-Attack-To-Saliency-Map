"""Plot attack approaches across epsilon from generated statistics.json files.

Run from the repository root:
    python -m evaluate_latex_generate.plot_epsilon_curves --input-dir result_latexs

Outputs epsilon_curves.png, epsilon_curves.pdf and curve_statistics.json under
<input-dir>/epsilon_curves (override with --output-dir).

Each seed is averaged equally over the available (model, explainer) cells.
Lines show the mean of these five seed averages; shading is +/- sample SD
(ddof=1), not the average of cell SDs or a confidence interval. H-score is
averaged from its recorded seed values, not recomputed from averaged ASR/SRO.
Missing table cells are not imputed. Coverage and seeds must match across
approaches and epsilon; different strategies/losses must be plotted separately.
Paper styling follows paretofront_compare_ui: hollow circles, serif text,
blue/teal/purple GASS curves and a thicker dark-red Ours curve.
Epsilon values use equally spaced categorical positions, not a linear budget axis.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


APPROACHES = {
    "ga_0": r"GASS ($\lambda=0.0$)",
    "ga_0.5": r"GASS ($\lambda=0.5$)",
    "ga_1": r"GASS ($\lambda=1.0$)",
    "nsgaii": r"$\mathbf{Ours}$",
}
PLOT_COLORS = {
    "ga_0": "#377EB8",
    "ga_0.5": "#269D86",
    "ga_1": "#8172B3",
    "nsgaii": "#9B1C31",
}
METRICS = {
    "asr": r"ASR $\uparrow$",
    "sro": r"SRO $\downarrow$",
    "h_score": r"H-score $\uparrow$",
}


def aggregate_reports(paths):
    points = {}
    configuration = None
    expected_cells = None
    expected_seeds = None
    for path in sorted(paths):
        report = json.loads(path.read_text(encoding="utf-8"))
        config = dict(report["configuration"])
        eps = config.pop("eps")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps):
            raise ValueError(f"{path}: epsilon must be a finite number")
        if eps in points:
            raise ValueError(f"{path}: duplicate epsilon {eps}")
        if configuration is None:
            configuration = config
        elif config != configuration:
            raise ValueError(f"{path}: incompatible configurations; separate strategies/losses")
        if report["averaging"] != "equal_weight_seed_means":
            raise ValueError(f"{path}: expected equal-weight seed statistics")

        grouped = {approach: {} for approach in APPROACHES}
        for cell in report["cells"]:
            approach = cell["approach"]
            if approach not in grouped:
                raise ValueError(f"{path}: unsupported approach {approach}")
            key = (cell["model"], cell["explain_method"])
            if key in grouped[approach]:
                raise ValueError(f"{path}: duplicate cell {approach}, {key}")
            grouped[approach][key] = cell["metrics"]

        points[eps] = {}
        for approach, cells in grouped.items():
            if not cells:
                raise ValueError(f"{path}: no cells for {approach}")
            if expected_cells is None:
                expected_cells = set(cells)
            elif set(cells) != expected_cells:
                raise ValueError(
                    f"{path}: model/explainer coverage differs for {approach}; "
                    "all approaches and epsilon must have identical cells"
                )
            metrics = {}
            for metric in METRICS:
                per_cell = []
                for key, values in cells.items():
                    seed_values = values[metric]["seed_values"]
                    seeds = set(seed_values)
                    if len(seeds) != 5:
                        raise ValueError(f"{path}: {approach}, {key}, {metric} needs five seeds")
                    if expected_seeds is None:
                        expected_seeds = seeds
                    elif seeds != expected_seeds:
                        raise ValueError(f"{path}: mismatched seeds for {approach}, {key}, {metric}")
                    if any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        for value in seed_values.values()
                    ):
                        raise ValueError(f"{path}: non-finite/non-numeric {metric} for {key}")
                    per_cell.append(seed_values)
                seed_means = {
                    seed: mean(values[seed] for values in per_cell)
                    for seed in sorted(expected_seeds)
                }
                metrics[metric] = {
                    "mean": mean(seed_means.values()),
                    "std": stdev(seed_means.values()),
                    "n_seeds": len(seed_means),
                    "seed_values": seed_means,
                }
            points[eps][approach] = metrics

    if len(points) < 2:
        raise ValueError("At least two epsilon statistics files are required")
    return {
        "configuration": configuration,
        "averaging": "equal_weight_model_explainer_cells_within_seed_then_equal_weight_seeds",
        "std": "sample_std_ddof_1_across_five_seed_means",
        "coverage": [
            {"model": model, "explain_method": method}
            for model, method in sorted(expected_cells)
        ],
        "points": [
            {"eps": eps, "approaches": points[eps]}
            for eps in sorted(points)
        ],
    }


def plot_curves(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with matplotlib.rc_context({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9,
        "axes.labelsize": 12, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), facecolor="white")
        epsilons = [point["eps"] for point in summary["points"]]
        positions = list(range(len(epsilons)))
        for ax, (metric, label) in zip(axes, METRICS.items()):
            ax.set_box_aspect(1)
            for approach, name in APPROACHES.items():
                color = PLOT_COLORS[approach]
                ours = approach == "nsgaii"
                stats = [point["approaches"][approach][metric] for point in summary["points"]]
                means = [stat["mean"] for stat in stats]
                lower = [stat["mean"] - stat["std"] for stat in stats]
                upper = [stat["mean"] + stat["std"] for stat in stats]
                ax.plot(
                    positions, means, label=name, color=color, marker="o",
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.4 if ours else 1.1,
                    linewidth=1.5 if ours else 0.9, markersize=4.5,
                    zorder=4 if ours else 3,
                )
                ax.fill_between(
                    positions, lower, upper, color=color, alpha=0.12,
                    linewidth=0, zorder=2,
                )
            ax.set_xlabel(r"$\epsilon$", labelpad=4)
            ax.set_ylabel(label, labelpad=4)
            ax.set_xticks(positions, labels=[f"{eps:g}" for eps in epsilons])
            ax.set_facecolor("white")
            ax.set_axisbelow(True)
            ax.grid(color="#DCE1E7", linewidth=0.55, linestyle="--", alpha=0.7)
            ax.tick_params(direction="in", top=True, right=True, width=0.7, length=3.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.75)
                spine.set_color("#414141")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=len(APPROACHES),
            frameon=False, fontsize=12, markerscale=1.3,
            handlelength=1.6, handletextpad=0.5,
        )
        fig.tight_layout(rect=(0, 0.12, 1, 1), pad=0.6, w_pad=1.0)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("result_latexs"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("attack_*/statistics.json"))
    summary = aggregate_reports(paths)
    figure = plot_curves(summary)
    output_dir = args.output_dir or args.input_dir / "epsilon_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    for extension in ("png", "pdf"):
        path = output_dir / f"epsilon_curves.{extension}"
        with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            figure.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        print(f"[OK] {path}")
    path = output_dir / "curve_statistics.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[OK] {path}")
    print(f"Coverage: {summary['coverage']}")
    import matplotlib.pyplot as plt

    plt.close(figure)


if __name__ == "__main__":
    main()
