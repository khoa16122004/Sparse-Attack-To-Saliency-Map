"""Interactive Pareto comparison; launch with streamlit run."""

import csv
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go
import streamlit as st

if __package__:
    from .paretofront_compare import (
        FrontComparison, RunMeta, _collect_run_fronts,
        _compare_fronts, _make_label, _parse_approach, _read_front,
    )
else:
    from paretofront_compare import (
        FrontComparison, RunMeta, _collect_run_fronts,
        _compare_fronts, _make_label, _parse_approach, _read_front,
    )


Point = Tuple[float, float]
Selection = Tuple[int, str, str]
PLOT_COLORS = {
    "ga_0.0_1.0": "#377EB8",
    "ga_0.5_0.5": "#269D86",
    "ga_1.0_0.0": "#8172B3",
    "nsgaii": "#9B1C31",
}


def _plot_order(keys: List[str]) -> List[str]:
    return sorted(keys, key=lambda key: (
        key == "nsgaii", float(key.split("_")[1]) if key != "nsgaii" else 0,
    ))


def _plot_label(key: str) -> str:
    if key == "nsgaii":
        return r"$\mathbf{Ours}$"
    wm = float(key.split("_")[1])
    return rf"GASS ($\lambda={wm:g}$)"


@dataclass
class SampleFronts:
    eps: int
    sample_id: str
    fronts: Dict[str, List[Point]]
    paths: Dict[str, Path]


@dataclass
class RankedComparison:
    sample: SampleFronts
    ga_key: str
    score: FrontComparison

    @property
    def selection(self) -> Selection:
        return self.sample.eps, self.sample.sample_id, self.ga_key

    @property
    def label(self) -> str:
        return (
            f"{self.score.advantage:+.0%} net | NSGA-II > GA {self.score.coverage:.0%}"
            f" | eps={self.sample.eps} | {self.sample.sample_id} | {_make_label(self.ga_key)}"
        )

    def row(self) -> dict:
        return {
            "Sample": self.sample.sample_id,
            "Epsilon": self.sample.eps,
            "GA": _make_label(self.ga_key),
            "NSGA-II > GA (%)": round(100 * self.score.coverage, 2),
            "GA > NSGA-II (%)": round(100 * self.score.reverse_coverage, 2),
            "Net advantage (pp)": round(100 * self.score.advantage, 2),
            "Dominated GA points": self.score.ga_dominated,
            "GA points": self.score.ga_count,
            "NSGA-II points": self.score.nsgaii_count,
        }


@st.cache_data(show_spinner=False)
def discover_runs(root: str, algorithm: str) -> List[RunMeta]:
    base = Path(root).expanduser()
    if not base.is_dir():
        raise ValueError(f"Not a directory: {base}")
    runs: List[RunMeta] = []
    for run_dir in sorted(base.glob("*/*/*")):
        if not run_dir.is_dir():
            continue
        meta = _parse_approach(run_dir.name)
        if meta is not None and meta.algorithm == algorithm:
            meta.run_dir = run_dir
            runs.append(meta)
    return runs


@st.cache_data(show_spinner=False)
def load_samples(
    selected: List[Tuple[int, str, str]],
) -> Tuple[List[SampleFronts], List[str]]:
    groups: Dict[int, Dict[str, Dict[str, Path]]] = {}
    for eps, key, run_dir in selected:
        groups.setdefault(eps, {})[key] = _collect_run_fronts(Path(run_dir))
    samples: List[SampleFronts] = []
    issues: List[str] = []
    for eps, run_maps in sorted(groups.items()):
        ns_map = run_maps.get("nsgaii", {})
        ga_keys = [key for key in run_maps if key != "nsgaii"]
        if not ns_map or not ga_keys:
            issues.append(f"eps={eps}: need NSGA-II fronts and at least one GA run.")
            continue
        for key in ga_keys:
            missing = set(ns_map) ^ set(run_maps[key])
            if missing:
                issues.append(
                    f"eps={eps}, {_make_label(key)}: {len(missing)} samples exist "
                    "on only one side; those pairs are excluded."
                )
        sample_ids = set(ns_map) & set().union(*(set(run_maps[k]) for k in ga_keys))
        for sample_id in sorted(sample_ids):
            fronts: Dict[str, List[Point]] = {}
            paths: Dict[str, Path] = {}
            for key, sample_map in run_maps.items():
                path = sample_map.get(sample_id)
                if path is None:
                    continue
                try:
                    points = _read_front(path, strict=True)
                except (ValueError, OSError, UnicodeError) as exc:
                    issues.append(f"{path}: {exc}")
                    continue
                if not points or any(not math.isfinite(v) for p in points for v in p):
                    issues.append(f"{path}: empty or non-finite front; excluded.")
                    continue
                fronts[key] = sorted(set(points))
                paths[key] = path
            if "nsgaii" in fronts and len(fronts) >= 2:
                samples.append(SampleFronts(eps, sample_id, fronts, paths))
    return samples, issues


def rank_comparisons(samples: List[SampleFronts]) -> List[RankedComparison]:
    ranked = [
        RankedComparison(sample, key, _compare_fronts(sample.fronts["nsgaii"], points))
        for sample in samples
        for key, points in sample.fronts.items()
        if key != "nsgaii"
    ]
    return sorted(ranked, key=lambda r: (
        -r.score.advantage, -r.score.coverage, -r.score.ga_dominated,
        r.sample.eps, r.sample.sample_id, r.ga_key,
    ))


def make_figure(
    sample: SampleFronts, visible: List[str], loss_type: str,
    marker_size: int, height: int, normalize: bool,
) -> go.Figure:
    figure = go.Figure()
    points = [p for key in visible for p in sample.fronts[key]]
    if not points:
        raise ValueError("Select at least one front to plot")
    minima = [min(p[i] for p in points) for i in (0, 1)]
    spans = [max(p[i] for p in points) - minima[i] for i in (0, 1)]
    for key in _plot_order(visible):
        front = sorted(sample.fronts[key])
        xs = [(p[0] - minima[0]) / (spans[0] or 1.0) if normalize else p[0] for p in front]
        ys = [(p[1] - minima[1]) / (spans[1] or 1.0) if normalize else p[1] for p in front]
        figure.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=_plot_label(key),
            line={
                "color": PLOT_COLORS.get(key, "#685547"),
                "width": 2 if key == "nsgaii" else 1.25, "shape": "linear",
            },
            marker={
                "symbol": "circle", "size": marker_size, "color": "white",
                "line": {
                    "width": 2 if key == "nsgaii" else 1.5,
                    "color": PLOT_COLORS.get(key, "#685547"),
                },
            },
            customdata=front,
            hovertemplate=(
                "Objective: %{customdata[0]:.6g}<br>"
                "Saliency: %{customdata[1]:.6g}<extra>%{fullData.name}</extra>"
            ),
        ))
    suffix = r"\;(\mathrm{min\!-\!max\ scaled})" if normalize else ""
    for axis, title, dim in (
        ("x", r"\mathcal{L}_{\text{attack}}", 0),
        ("y", r"\mathcal{L}_{\text{explain}}", 1),
    ):
        low = 0.0 if normalize else minima[dim]
        high = (1.0 if spans[dim] else 0.0) if normalize else low + spans[dim]
        padding = (high - low) * 0.04 or max(abs(low) * 0.02, 0.01)
        figure.update_layout(**{f"{axis}axis": {
            "title": f"${title}{suffix}$", "range": [low - padding, high + padding],
            "showgrid": True, "gridcolor": "#E9EEF5", "zeroline": False,
            "showline": True, "linecolor": "#CAD3DF", "mirror": True,
            "exponentformat": "power",
        }})
    figure.update_layout(
        template="plotly_white", height=height,
        margin={"l": 65, "r": 25, "t": 25, "b": 55},
        font={"family": "Arial, sans-serif", "size": 13, "color": "#26354A"},
        legend={
            "orientation": "v", "y": 0.98, "x": 0.98,
            "xanchor": "right", "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.9)",
        },
        hovermode="closest", dragmode="zoom",
    )
    return figure


def make_paper_figure(
    sample: SampleFronts, visible: List[str], loss_type: str,
    marker_size: int, width: int, height: int, normalize: bool,
) -> Figure:
    # Reuse the interactive plot's coordinates and limits so both views agree.
    visible = _plot_order(visible)
    source = make_figure(sample, visible, loss_type, marker_size, height, normalize)
    with mpl.rc_context({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.labelsize": 16, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        figure = Figure(figsize=(width / 120, height / 120), dpi=150, facecolor="white")
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        ax.set_box_aspect(1)
        for key, trace in zip(visible, source.data):
            ax.plot(
                trace.x, trace.y, marker="o", markersize=marker_size / math.sqrt(2),
                markerfacecolor="white", markeredgecolor=trace.line.color,
                markeredgewidth=1.4 if key == "nsgaii" else 1.1,
                color=trace.line.color, linewidth=1.5 if key == "nsgaii" else 0.9,
                label=_plot_label(key), zorder=4 if key == "nsgaii" else 3,
            )
        ax.set_xlabel(source.layout.xaxis.title.text, labelpad=7)
        ax.set_ylabel(source.layout.yaxis.title.text, labelpad=7)
        ax.set_xlim(source.layout.xaxis.range)
        ax.set_ylim(source.layout.yaxis.range)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10], min_n_ticks=2))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10], min_n_ticks=2))
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        ax.grid(color="#DCE1E7", linewidth=0.55, linestyle="--", alpha=0.7)
        ax.tick_params(direction="in", top=True, right=True, width=0.7, length=3.5)
        ax.ticklabel_format(style="sci", scilimits=(-3, 4), useMathText=True, useOffset=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
            spine.set_color("#414141")
        legend = ax.legend(
            loc="upper right", ncol=1, fontsize=10,
            frameon=True, facecolor="white", edgecolor="none", framealpha=0.9,
            handlelength=1.4, handletextpad=0.5, labelspacing=0.45,
            borderpad=0.55, borderaxespad=0.5,
        )
        legend.set_zorder(10)
        figure.tight_layout(pad=0.7)
    return figure


def export_paper_figure(figure: Figure, file_format: str) -> bytes:
    buffer = io.BytesIO()
    with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path"}):
        figure.savefig(buffer, format=file_format, dpi=600, facecolor="white")
    return buffer.getvalue()


def _move_selection(options: List[Selection], offset: int) -> None:
    current = options.index(st.session_state["comparison"])
    st.session_state["comparison"] = options[(current + offset) % len(options)]


def _select_runs(runs: List[RunMeta]) -> List[Tuple[int, str, str]]:
    by_key: Dict[Tuple[int, str], List[RunMeta]] = {}
    for run in runs:
        if run.algorithm == "ga":
            if run.w_m is None or run.w_s is None:
                continue
            key = f"ga_{run.w_m}_{run.w_s}"
        else:
            key = "nsgaii"
        by_key.setdefault((run.eps, key), []).append(run)
    eps_options = sorted({eps for eps, _ in by_key})
    eps_selected = st.sidebar.multiselect("Epsilon", eps_options, default=eps_options)
    ga_options = sorted({key for _, key in by_key if key != "nsgaii"})
    ga_selected = st.sidebar.multiselect(
        "GA weight pairs", ga_options, default=ga_options, format_func=_make_label,
    )
    selected: List[Tuple[int, str, str]] = []
    for (eps, key), candidates in sorted(by_key.items()):
        if eps not in eps_selected or (key != "nsgaii" and key not in ga_selected):
            continue
        if len(candidates) > 1:
            chosen = st.sidebar.selectbox(
                f"Run: eps={eps}, {_make_label(key)}",
                [str(c.run_dir) for c in candidates],
                format_func=lambda path: Path(path).name,
            )
            st.sidebar.caption("Multiple matching runs: choose explicitly; no fronts are merged.")
        else:
            chosen = str(candidates[0].run_dir)
        selected.append((eps, key, chosen))
    return selected


def main() -> None:
    st.set_page_config(page_title="Pareto Front Explorer", page_icon="◎", layout="wide")
    st.title("Pareto Front Explorer")
    st.caption("NSGA-II vs weighted-sum GA | per-sample inspection | both objectives minimized")
    with st.sidebar:
        st.header("Experiment")
        ga_root = st.text_input("GA root (contains seed / model / run)", "D:\\ESWA\\server_run_seed\\GA")
        ns_root = st.text_input("NSGA-II root", "D:\\ESWA\\server_run_seed\\NSGAII")
        if st.button("Reload files", help="Clear cached directory scans and front scores."):
            discover_runs.clear()
            load_samples.clear()
    try:
        runs = discover_runs(ga_root, "ga") + discover_runs(ns_root, "nsgaii")
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        st.info("Set the two algorithm roots in the sidebar, then click Reload files.")
        return
    if not runs:
        st.warning("No recognized runs found. Expected: algorithm root / seed / model / approach.")
        return
    for title, getter in (
        ("Seed", lambda r: r.run_dir.parent.parent.name),
        ("Model", lambda r: r.run_dir.parent.name),
        ("Strategy", lambda r: r.strategy),
        ("Explanation", lambda r: r.explain_method),
        ("Loss", lambda r: r.loss_type),
    ):
        value = st.sidebar.selectbox(title, sorted({getter(r) for r in runs}))
        runs = [r for r in runs if getter(r) == value]
    loss_type = runs[0].loss_type
    selected = _select_runs(runs)
    with st.spinner("Reading fronts and computing strict Pareto coverage..."):
        samples, issues = load_samples(selected)
        ranked = rank_comparisons(samples)
    if issues:
        with st.expander(f"Data warnings ({len(issues)})"):
            for issue in issues:
                st.warning(issue)
    st.caption(
        "Only matching seed/model/strategy/explanation/loss/epsilon and sample are compared. "
        "No margin-to-cross-entropy fallback: stored objective scores are not interchangeable."
    )
    if not ranked:
        st.warning("No valid NSGA-II / GA sample pairs. Check filters and front files.")
        return
    with st.sidebar:
        st.header("Find strong NSGA-II wins")
        min_coverage = st.slider("Minimum GA points dominated (%)", 0, 100, 0, 5)
        max_reverse = st.slider("Maximum NSGA-II points dominated (%)", 0, 100, 100, 5)
        search = st.text_input("Search class / sample")
        st.header("Plot")
        plot_style = st.radio("View", ["Paper figure", "Interactive"])
        if plot_style == "Paper figure":
            width = st.slider("Square figure size (px)", 360, 720, 380, 20)
            height = width
            st.caption(f"Square export: {width / 120:.2f} inches per side; tight data limits.")
        else:
            height = st.slider("Plot height (px)", 300, 700, 420, 20)
            width = st.slider("Plot width (px)", 440, 1200, 760, 40)
        marker_size = st.slider("Hollow marker size", 4, 16, 8)
        normalize = st.checkbox("Min-max scale axes", help="Plot only; ranking uses raw scores.")
    filtered = [
        r for r in ranked
        if 100 * r.score.coverage >= min_coverage
        and 100 * r.score.reverse_coverage <= max_reverse
        and search.casefold() in r.sample.sample_id.casefold()
    ]
    if not filtered:
        st.info("No pairs match these dominance thresholds. Relax the filters in the sidebar.")
        return
    st.subheader("Strongest combinations")
    st.caption(
        f"{len(filtered)} / {len(ranked)} valid pairs. Sorted by net advantage, then coverage. "
        "This is a filtered case ranking, not an overall algorithm performance claim."
    )
    rows = [r.row() for r in filtered]
    st.dataframe(rows[:30], hide_index=True, use_container_width=True)
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    st.download_button("Download filtered ranking (CSV)", csv_buffer.getvalue(), "pareto_ranking.csv", "text/csv")

    by_selection = {r.selection: r for r in filtered}
    options = list(by_selection)
    if st.session_state.get("comparison") not in by_selection:
        st.session_state["comparison"] = options[0]
    prev, picker, nxt = st.columns([1, 8, 1])
    with prev:
        st.button("Previous", on_click=_move_selection, args=(options, -1))
    with picker:
        selection = st.selectbox(
            "Sample / comparison", options, key="comparison",
            format_func=lambda option: by_selection[option].label,
        )
    with nxt:
        st.button("Next", on_click=_move_selection, args=(options, 1))
    current = by_selection[selection]
    score = current.score
    columns = st.columns(3)
    columns[0].metric("GA points dominated by NSGA-II", f"{score.coverage:.1%}", f"{score.ga_dominated}/{score.ga_count} points", delta_color="off")
    columns[1].metric("NSGA-II points dominated by GA", f"{score.reverse_coverage:.1%}", f"{score.nsgaii_dominated}/{score.nsgaii_count} points", delta_color="off")
    columns[2].metric("Net coverage advantage", f"{100 * score.advantage:+.1f} pp")
    st.subheader(f"{current.sample.sample_id} | epsilon={current.sample.eps}")
    visible = st.multiselect(
        "Visible fronts", list(current.sample.fronts),
        default=list(current.sample.fronts), format_func=_make_label,
    )
    if visible:
        if plot_style == "Paper figure":
            paper = make_paper_figure(
                current.sample, visible, loss_type, marker_size, width, height, normalize,
            )
            with st.expander("Crop / zoom for export"):
                if st.checkbox("Custom axis limits"):
                    ax = paper.axes[0]
                    xlow, xhigh = ax.get_xlim()
                    ylow, yhigh = ax.get_ylim()
                    bounds = st.columns(4)
                    xmin = bounds[0].number_input("X min", value=float(xlow), format="%.6g")
                    xmax = bounds[1].number_input("X max", value=float(xhigh), format="%.6g")
                    ymin = bounds[2].number_input("Y min", value=float(ylow), format="%.6g")
                    ymax = bounds[3].number_input("Y max", value=float(yhigh), format="%.6g")
                    if xmin >= xmax or ymin >= ymax:
                        st.error("Each axis minimum must be smaller than its maximum.")
                        return
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
            png = export_paper_figure(paper, "png")
            st.image(png, width=width)
            st.caption("Paper-ready figure: no sample title; raw loss axes by default. PNG 600 dpi; PDF/SVG are vector.")
            filename = f"pareto_eps{current.sample.eps}_{current.sample.sample_id.replace('/', '__')}"
            buttons = st.columns(3)
            for column, fmt, mime in zip(
                buttons, ("png", "pdf", "svg"),
                ("image/png", "application/pdf", "image/svg+xml"),
            ):
                column.download_button(
                    f"Download {fmt.upper()}", png if fmt == "png" else export_paper_figure(paper, fmt),
                    f"{filename}.{fmt}", mime,
                )
        else:
            figure = make_figure(current.sample, visible, loss_type, marker_size, height, normalize)
            figure.update_layout(width=width)
            st.plotly_chart(figure, use_container_width=False, theme=None, config={
                "scrollZoom": True, "displaylogo": False,
                "toImageButtonOptions": {"format": "png", "filename": "pareto_front", "scale": 3},
            })
            st.caption("Drag a box or scroll to zoom; double-click to reset. Click legend entries to toggle fronts.")
            st.download_button(
                "Download interactive plot (HTML)",
                figure.to_html(include_plotlyjs=True).encode("utf-8"),
                "pareto_front.html", "text/html",
            )
    else:
        st.info("Select at least one front to plot.")
    with st.expander("Definition and source files"):
        st.markdown(
            "A strictly dominates B if A is no worse on both losses and strictly better on "
            "at least one. Equal points do **not** dominate each other. Duplicate coordinates "
            "are counted once. Coverage is the fraction of the target front dominated by "
            "at least one source point; net advantage = forward coverage - reverse coverage. "
            "It is **not** hypervolume or the number of dominating point pairs."
        )
        for key, path in current.sample.paths.items():
            st.text(f"{_make_label(key)}: {path}")


if __name__ == "__main__":
    main()
