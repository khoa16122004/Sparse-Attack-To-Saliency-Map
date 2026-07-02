import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate batch attack outputs with mean Attack Success Rate, "
            "Margin Loss, Saliency Loss, and Spearman rank correlation"
        )
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Path to batch_report.json produced by run_batch.py",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help=(
            "Path to directory containing multiple run folders. "
            "Each run folder should have batch_report.json"
        ),
    )
    parser.add_argument(
        "--print-latex-lines",
        action="store_true",
        help=(
            "Print one LaTeX row per epsilon in format: "
            "& eps & margin_asr & margin_spearman & ce_asr & ce_spearman \\\\"
        ),
    )
    parser.add_argument(
        "--print-markdown-tables",
        action="store_true",
        help=(
            "Print Markdown comparison tables grouped by epsilon. "
            "Rows are approach variants (strategy + explain method + fitness + algo)."
        ),
    )
    parser.add_argument(
        "--markdown-output-file",
        type=str,
        default=None,
        help=(
            "Path to save Markdown comparison tables. "
            "Works with --report-dir and --print-markdown-tables."
        ),
    )
    parser.add_argument(
        "--failed-output-file",
        type=str,
        default=None,
        help=(
            "Path to save failure report JSON in --report-dir mode. "
            "Contains failed runs and Spearman-failed samples."
        ),
    )
    parser.add_argument(
        "--filled-output-file",
        type=str,
        default=None,
        help=(
            "Path to save an auto-filled Markdown summary table in --report-dir mode. "
            "Supports filtering by --include-models and --include-algorithms."
        ),
    )
    parser.add_argument(
        "--filled-output-format",
        type=str,
        default="markdown",
        choices=["markdown", "latex"],
        help="Format for --filled-output-file: markdown (default) or latex",
    )
    parser.add_argument(
        "--include-models",
        type=str,
        default=None,
        help="Comma-separated model names to include in filled output (e.g. resnet50,vgg16)",
    )
    parser.add_argument(
        "--include-algorithms",
        type=str,
        default=None,
        help="Comma-separated algorithms to include in filled output (e.g. weighted_sum_ga,nsgaii)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only first N successful samples",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Where to save evaluation JSON for single mode. Default: <run_root>/evaluation_summary.json",
    )
    return parser.parse_args()


def _extract_eps_and_loss_type(approach):
    if not approach:
        return None, None

    eps_match = re.search(r"__eps-([0-9]+(?:\.[0-9]+)?)__", approach)
    if not eps_match:
        return None, None

    eps_value = float(eps_match.group(1))
    if eps_value.is_integer():
        eps_value = int(eps_value)

    if "fit-negative_cross_entropy_saliency" in approach:
        loss_type = "cross_entropy"
    else:
        loss_type = "margin"

    return eps_value, loss_type


def _extract_approach_fields(approach):
    fields = {
        "strategy": "unknown",
        "explain_method": "unknown",
        "fitness_name": "margin_default",
        "algo": "default",
        "eps": None,
    }
    if not approach:
        return fields

    for token in str(approach).split("__"):
        if token.startswith("strategy-"):
            fields["strategy"] = token[len("strategy-") :]
        elif token.startswith("exp-"):
            fields["explain_method"] = token[len("exp-") :]
        elif token.startswith("fit-"):
            fields["fitness_name"] = token[len("fit-") :]
        elif token.startswith("algo-"):
            fields["algo"] = token[len("algo-") :]
        elif token.startswith("eps-"):
            try:
                eps_value = float(token[len("eps-") :])
                fields["eps"] = int(eps_value) if eps_value.is_integer() else eps_value
            except ValueError:
                pass

    return fields


def _parse_csv_filter(raw_value):
    if raw_value is None:
        return None

    values = [part.strip() for part in str(raw_value).split(",") if part.strip()]
    if not values:
        return None
    return set(values)


def _normalize_algorithm_name(algo_name):
    normalized = str(algo_name or "").strip().lower()
    if normalized in {"", "default", "weighted_sum", "wga"}:
        return "weighted_sum_ga"
    return normalized


def _format_asr_percent(asr):
    return f"{(float(asr) * 100.0):.2f}"


def _format_spearman(score):
    return f"{float(score):.4f}"


def _latex_bold(text, should_bold):
    if should_bold:
        return f"\\textbf{{{text}}}"
    return text


def _print_latex_rows_from_report_dir(report_dir):
    root = Path(report_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"report directory not found: {root}")

    rows_by_eps = {}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        summary_path = run_dir / "evaluation_summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        approach = summary.get("approach", "")
        eps_value, loss_type = _extract_eps_and_loss_type(approach)
        if eps_value is None or loss_type is None:
            report_path = run_dir / "batch_report.json"
            if report_path.exists():
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                approach = report.get("approach", "")
                eps_value, loss_type = _extract_eps_and_loss_type(approach)

        if eps_value is None or loss_type is None:
            continue

        rows_by_eps.setdefault(eps_value, {})
        if loss_type in rows_by_eps[eps_value]:
            # Keep first one in sorted order for deterministic output.
            continue

        rows_by_eps[eps_value][loss_type] = {
            "asr": float(summary.get("attack_success_rate", float("nan"))),
            "spearman": float(summary.get("mean_spearman_adv_vs_clean_saliency", float("nan"))),
        }

    if not rows_by_eps:
        print("[WARN] No evaluation_summary/batch_report with parseable epsilon in approach")
        return

    printed = 0
    for eps_value in sorted(rows_by_eps.keys(), key=float):
        pair = rows_by_eps[eps_value]
        margin = pair.get("margin")
        cross_entropy = pair.get("cross_entropy")
        if margin is None or cross_entropy is None:
            continue

        margin_asr = _format_asr_percent(margin["asr"])
        margin_spearman = _format_spearman(margin["spearman"])
        ce_asr = _format_asr_percent(cross_entropy["asr"])
        ce_spearman = _format_spearman(cross_entropy["spearman"])

        margin_asr_val = float(margin_asr)
        ce_asr_val = float(ce_asr)
        margin_spearman_val = float(margin_spearman)
        ce_spearman_val = float(ce_spearman)

        margin_asr_txt = _latex_bold(margin_asr, margin_asr_val >= ce_asr_val)
        ce_asr_txt = _latex_bold(ce_asr, ce_asr_val >= margin_asr_val)
        margin_spearman_txt = _latex_bold(
            margin_spearman, margin_spearman_val <= ce_spearman_val
        )
        ce_spearman_txt = _latex_bold(
            ce_spearman, ce_spearman_val <= margin_spearman_val
        )

        print(
            f"& {eps_value} & {margin_asr_txt} & {margin_spearman_txt} "
            f"& {ce_asr_txt} & {ce_spearman_txt} \\\\"
        )
        printed += 1

    if printed == 0:
        print("[WARN] No complete margin/cross_entropy pairs for LaTeX rows")


def _format_float(value, decimals=4, multiply_100=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"

    if math.isnan(number):
        return "NaN"

    if multiply_100:
        number *= 100.0

    return f"{number:.{decimals}f}"


def _latex_escape(text):
    value = str(text)
    replacements = {
        "\\": r"\\textbackslash{}",
        "&": r"\\&",
        "%": r"\\%",
        "$": r"\\$",
        "#": r"\\#",
        "_": r"\\_",
        "{": r"\\{",
        "}": r"\\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _build_filtered_filled_rows(evaluated, include_models=None, include_algorithms=None):
    include_models = set(include_models or [])
    include_algorithms = {_normalize_algorithm_name(name) for name in (include_algorithms or [])}

    rows = []
    for run_dir, summary, _ in evaluated:
        model_name = str(summary.get("model", "unknown"))
        approach = str(summary.get("approach", ""))
        fields = _extract_approach_fields(approach)
        algorithm = _normalize_algorithm_name(fields.get("algo"))
        eps = fields.get("eps")
        if eps is None:
            eps, _ = _extract_eps_and_loss_type(approach)

        if include_models and model_name not in include_models:
            continue
        if include_algorithms and algorithm not in include_algorithms:
            continue

        rows.append(
            {
                "model": model_name,
                "algorithm": algorithm,
                "strategy": fields.get("strategy", "unknown"),
                "explain_method": fields.get("explain_method", "unknown"),
                "fitness": fields.get("fitness_name", "margin_default"),
                "eps": eps,
                "asr": summary.get("attack_success_rate", float("nan")),
                "spearman": summary.get("mean_spearman_adv_vs_clean_saliency", float("nan")),
                "margin_loss": summary.get("mean_margin_loss", float("nan")),
                "saliency_loss": summary.get("mean_saliency_loss", float("nan")),
                "ok_samples": summary.get("num_ok_samples_evaluated", 0),
                "spearman_failed": summary.get("spearman_failed_samples", 0),
                "run_dir": str(run_dir),
            }
        )

    rows.sort(
        key=lambda x: (
            x["model"],
            x["algorithm"],
            float(x["eps"]) if x["eps"] is not None else float("inf"),
            x["strategy"],
            x["explain_method"],
            x["fitness"],
            x["run_dir"],
        )
    )
    return rows


def _render_filled_markdown(rows, report_dir, include_models=None, include_algorithms=None):
    lines = []
    lines.append("# Auto-filled Evaluation Summary")
    lines.append(f"- report_dir: {report_dir}")
    lines.append(
        "- include_models: "
        + (", ".join(sorted(include_models)) if include_models else "ALL")
    )
    lines.append(
        "- include_algorithms: "
        + (", ".join(sorted(include_algorithms)) if include_algorithms else "ALL")
    )
    lines.append("")

    if not rows:
        lines.append("[WARN] No evaluated rows matched the selected model/algorithm filters.")
        return "\n".join(lines)

    lines.append(
        "| Model | Algorithm | Eps | Strategy | Explain | Fitness | ASR (%) | Spearman | "
        "Margin Loss | Saliency Loss | OK Samples | Spearman Failed | Run Dir |"
    )
    lines.append("|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|")

    for row in rows:
        eps_text = "-" if row["eps"] is None else str(row["eps"])
        lines.append(
            "| "
            f"{row['model']} | "
            f"{row['algorithm']} | "
            f"{eps_text} | "
            f"{row['strategy']} | "
            f"{row['explain_method']} | "
            f"{row['fitness']} | "
            f"{_format_float(row['asr'], decimals=2, multiply_100=True)} | "
            f"{_format_float(row['spearman'], decimals=4)} | "
            f"{_format_float(row['margin_loss'], decimals=4)} | "
            f"{_format_float(row['saliency_loss'], decimals=4)} | "
            f"{row['ok_samples']} | "
            f"{row['spearman_failed']} | "
            f"{row['run_dir']} |"
        )

    return "\n".join(lines)


def _render_filled_latex(rows, report_dir, include_models=None, include_algorithms=None):
    def _strategy_label(strategy_name):
        mapping = {
            "saliency_guided": "Saliency-guided",
            "uniform": "Uniform",
        }
        return mapping.get(strategy_name, strategy_name)

    def _fit_bucket(fitness_name):
        if fitness_name == "negative_cross_entropy_saliency":
            return "negative_cross_entropy"
        return "margin"

    # model -> eps -> strategy -> bucket -> metrics
    grouped = {}
    for row in rows:
        model = str(row["model"])
        eps = row["eps"]
        strategy = str(row["strategy"])
        bucket = _fit_bucket(str(row["fitness"]))

        grouped.setdefault(model, {})
        grouped[model].setdefault(eps, {})
        grouped[model][eps].setdefault(strategy, {})

        if bucket not in grouped[model][eps][strategy]:
            grouped[model][eps][strategy][bucket] = {
                "asr": row["asr"],
                "spearman": row["spearman"],
            }

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{GA, $\\lambda=0.5, |P|=50, N=200, p_c=0.4, p_m=0.1$}"
    )
    lines.append("\\label{tab:placeholder}")
    lines.append("")
    lines.append("\\begin{tabular}{ccc|cc|cc}")
    lines.append("\\toprule")
    lines.append("\\multirow{2}{*}{Model} &")
    lines.append("\\multirow{2}{*}{$\\epsilon$} &")
    lines.append("\\multirow{2}{*}{Initialization} &")
    lines.append("\\multicolumn{2}{c|}{\\textbf{Margin Loss}} &")
    lines.append("\\multicolumn{2}{c}{\\textbf{Negative-Cross-Entropy}} \\\\")
    lines.append("\\cmidrule(lr){4-5}")
    lines.append("\\cmidrule(lr){6-7}")
    lines.append("&")
    lines.append("&")
    lines.append("&")
    lines.append("ASR$\\uparrow$ &")
    lines.append("SRO$\\downarrow$ &")
    lines.append("ASR$\\uparrow$ &")
    lines.append("SRO$\\downarrow$ \\\\")
    lines.append("\\midrule")
    lines.append("")

    if not grouped:
        lines.append("\\multicolumn{7}{c}{No evaluated rows matched selected filters} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table*}")
        return "\n".join(lines)

    strategy_order = {
        "saliency_guided": 0,
        "uniform": 1,
    }

    model_names = sorted(grouped.keys())
    for model_idx, model_name in enumerate(model_names):
        eps_values = sorted(
            grouped[model_name].keys(),
            key=lambda x: float(x) if x is not None else float("inf"),
        )

        model_row_count = 0
        for eps in eps_values:
            model_row_count += max(1, len(grouped[model_name][eps]))

        model_cell_consumed = False
        for eps_idx, eps in enumerate(eps_values):
            strategies = sorted(
                grouped[model_name][eps].keys(),
                key=lambda s: (strategy_order.get(s, 99), s),
            )
            if not strategies:
                strategies = ["unknown"]

            eps_row_count = len(strategies)
            eps_cell_consumed = False

            for strategy in strategies:
                fit_data = grouped[model_name][eps].get(strategy, {})
                margin_data = fit_data.get("margin", {})
                ce_data = fit_data.get("negative_cross_entropy", {})

                model_cell = ""
                eps_cell = ""
                if not model_cell_consumed:
                    model_cell = f"\\multirow{{{model_row_count}}}{{*}}{{{_latex_escape(model_name)}}}"
                    model_cell_consumed = True
                if not eps_cell_consumed:
                    eps_text = "-" if eps is None else str(eps)
                    eps_cell = f"\\multirow{{{eps_row_count}}}{{*}}{{{_latex_escape(eps_text)}}}"
                    eps_cell_consumed = True

                margin_asr = _format_float(margin_data.get("asr", float("nan")), decimals=2, multiply_100=True)
                margin_sro = _format_float(margin_data.get("spearman", float("nan")), decimals=4)
                ce_asr = _format_float(ce_data.get("asr", float("nan")), decimals=2, multiply_100=True)
                ce_sro = _format_float(ce_data.get("spearman", float("nan")), decimals=4)

                lines.append(
                    f"{model_cell} & {eps_cell} & {_latex_escape(_strategy_label(strategy))} "
                    f"& {_latex_escape(margin_asr)} & {_latex_escape(margin_sro)} "
                    f"& {_latex_escape(ce_asr)} & {_latex_escape(ce_sro)} \\\\"
                )

            if eps_idx != len(eps_values) - 1:
                lines.append("\\cmidrule(lr){2-7}")
                lines.append("")

        if model_idx != len(model_names) - 1:
            lines.append("\\midrule")
            lines.append("")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def _save_filled_output(
    report_dir,
    evaluated,
    filled_output_file,
    include_models=None,
    include_algorithms=None,
    output_format="markdown",
):
    rows = _build_filtered_filled_rows(
        evaluated=evaluated,
        include_models=include_models,
        include_algorithms=include_algorithms,
    )
    if output_format == "latex":
        filled_text = _render_filled_latex(
            rows=rows,
            report_dir=report_dir,
            include_models=include_models,
            include_algorithms=include_algorithms,
        )
    else:
        filled_text = _render_filled_markdown(
            rows=rows,
            report_dir=report_dir,
            include_models=include_models,
            include_algorithms=include_algorithms,
        )

    output_path = Path(filled_output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(filled_text + "\n")
    return output_path, len(rows)


def _print_markdown_tables_from_report_dir(report_dir):
    root = Path(report_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"report directory not found: {root}")

    lines = []

    rows_by_eps = {}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        summary_path = run_dir / "evaluation_summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        approach = str(summary.get("approach", ""))
        fields = _extract_approach_fields(approach)
        eps = fields["eps"]
        if eps is None:
            eps, _ = _extract_eps_and_loss_type(approach)

        if eps is None:
            continue

        rows_by_eps.setdefault(eps, [])
        rows_by_eps[eps].append(
            {
                "strategy": fields["strategy"],
                "explain_method": fields["explain_method"],
                "fitness_name": fields["fitness_name"],
                "algo": fields["algo"],
                "asr": summary.get("attack_success_rate", float("nan")),
                "spearman": summary.get("mean_spearman_adv_vs_clean_saliency", float("nan")),
                "margin_loss": summary.get("mean_margin_loss", float("nan")),
                "saliency_loss": summary.get("mean_saliency_loss", float("nan")),
                "ok_samples": summary.get("num_ok_samples_evaluated", 0),
                "run": run_dir.name,
            }
        )

    if not rows_by_eps:
        lines.append("[WARN] No evaluation_summary with parseable epsilon for Markdown tables")
        return "\n".join(lines)

    for eps in sorted(rows_by_eps.keys(), key=float):
        lines.append(f"\n## Epsilon = {eps}")
        lines.append(
            "| Strategy | Explain Method | Fitness | Algo | ASR (%) | Spearman | "
            "Margin Loss | Saliency Loss | OK Samples | Run |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|")

        rows = sorted(
            rows_by_eps[eps],
            key=lambda x: (
                x["strategy"],
                x["explain_method"],
                x["fitness_name"],
                x["algo"],
                x["run"],
            ),
        )

        for row in rows:
            lines.append(
                "| "
                f"{row['strategy']} | "
                f"{row['explain_method']} | "
                f"{row['fitness_name']} | "
                f"{row['algo']} | "
                f"{_format_float(row['asr'], decimals=2, multiply_100=True)} | "
                f"{_format_float(row['spearman'], decimals=4)} | "
                f"{_format_float(row['margin_loss'], decimals=4)} | "
                f"{_format_float(row['saliency_loss'], decimals=4)} | "
                f"{row['ok_samples']} | "
                f"{row['run']} |"
            )

    return "\n".join(lines)


def _rankdata_average_ties(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]

    n = values.shape[0]
    ranks_sorted = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks_sorted[i:j] = avg_rank
        i = j

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _spearman_rank_corr(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("Spearman inputs must have the same number of elements")
    if x.size < 2:
        return float("nan")

    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = math.sqrt(float(np.dot(rx, rx)) * float(np.dot(ry, ry)))
    if denom < 1e-12:
        return float("nan")

    return float(np.dot(rx, ry) / denom)


def _load_gray_image_array(image_path):
    image = Image.open(image_path).convert("L")
    return np.asarray(image, dtype=np.float64)


def _safe_mean(values):
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def _resolve_output_dir_candidates(result, report_path):
    report_parent = Path(report_path).parent
    raw_output_dir = result.get("output_dir", "")

    candidates = []
    if raw_output_dir:
        path = Path(str(raw_output_dir))
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(Path.cwd() / path)
            candidates.append(report_parent / path)
            repo_root = Path(__file__).resolve().parent.parent
            candidates.append(repo_root / path)

            # Rebuild using report ancestors when raw path starts with folder name
            # from a lower level, e.g. compare_loss_50/resnet50/... without server_result.
            if path.parts:
                first = path.parts[0]
                for ancestor in [report_parent] + list(report_parent.parents):
                    if ancestor.name == first:
                        candidates.append(ancestor.parent / path)

            # Fallback: keep only sample leaf and attach to current run folder.
            if len(path.parts) >= 2:
                candidates.append(report_parent / path.parts[-2] / path.parts[-1])

    # Most reliable source: sample folder under current run dir.
    class_name = result.get("class")
    image_ref = (
        result.get("resolved_image")
        or result.get("input_raw")
        or result.get("image")
        or result.get("img")
    )
    if class_name and image_ref:
        image_stem = Path(str(image_ref)).stem
        if image_stem:
            candidates.append(report_parent / str(class_name) / image_stem)

    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _compute_spearman_for_sample(result, report_path):
    candidates = _resolve_output_dir_candidates(result=result, report_path=report_path)
    tried_pairs = []

    clean_map_path = None
    adv_map_path = None
    selected_output_dir = None

    for output_dir in candidates:
        clean_candidate = output_dir / "clean_map.png"
        adv_candidate = output_dir / "adv_map.png"
        tried_pairs.append(
            {
                "clean_map_path": str(clean_candidate),
                "adv_map_path": str(adv_candidate),
            }
        )
        if clean_candidate.exists() and adv_candidate.exists():
            selected_output_dir = output_dir
            clean_map_path = clean_candidate
            adv_map_path = adv_candidate
            break

    if clean_map_path is None or adv_map_path is None:
        raw_output_dir = Path(str(result.get("output_dir", "")))
        return None, {
            "reason": "missing_clean_or_adv_map",
            "output_dir": str(raw_output_dir),
            "clean_map_path": str(raw_output_dir / "clean_map.png"),
            "adv_map_path": str(raw_output_dir / "adv_map.png"),
            "tried_paths": tried_pairs,
        }

    clean_map = _load_gray_image_array(clean_map_path)
    adv_map = _load_gray_image_array(adv_map_path)

    corr = _spearman_rank_corr(clean_map, adv_map)
    if corr is None or math.isnan(corr):
        return corr, {
            "reason": "nan_spearman",
            "resolved_output_dir": str(selected_output_dir),
            "clean_map_path": str(clean_map_path),
            "adv_map_path": str(adv_map_path),
        }

    return corr, None


def _evaluate_single_report(report_path, max_samples=None):
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"report file not found: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    model_name = report.get("model")

    ok_results = [item for item in report.get("results", []) if item.get("status") == "ok"]
    if max_samples is not None:
        ok_results = ok_results[:max_samples]

    if not ok_results:
        raise ValueError("No successful samples (status='ok') found in report")

    margin_losses = []
    saliency_losses = []
    success_flags = []
    spearman_scores = []
    spearman_failed = 0
    spearman_failed_details = []

    for result in tqdm(ok_results, desc=f"Evaluating {report_path.parent.name}", unit="img"):
        true_label = int(result.get("true_label", result.get("clean_pred", -1)))
        adv_pred = int(result.get("adv_pred", -1))

        margin_losses.append(float(result.get("margin_loss", float("nan"))))
        saliency_losses.append(float(result.get("saliency_loss", float("nan"))))
        success_flags.append(int(adv_pred != true_label and true_label >= 0 and adv_pred >= 0))

        corr, err = _compute_spearman_for_sample(result=result, report_path=report_path)
        if err is not None:
            spearman_failed += 1
            detail = dict(err)
            if "output_dir" not in detail:
                detail["output_dir"] = str(result.get("output_dir", ""))
            detail["image"] = result.get("img")
            detail["true_label"] = result.get("true_label", result.get("clean_pred"))
            detail["adv_pred"] = result.get("adv_pred")
            spearman_failed_details.append(detail)
        spearman_scores.append(corr)

    attack_success_rate = float(np.mean(success_flags))
    mean_margin_loss = _safe_mean(margin_losses)
    mean_saliency_loss = _safe_mean(saliency_losses)
    mean_spearman = _safe_mean(spearman_scores)

    return {
        "report_path": str(report_path),
        "model": model_name,
        "approach": report.get("approach"),
        "num_ok_samples_evaluated": len(ok_results),
        "attack_success_rate": attack_success_rate,
        "mean_margin_loss": mean_margin_loss,
        "mean_saliency_loss": mean_saliency_loss,
        "mean_spearman_adv_vs_clean_saliency": mean_spearman,
        "spearman_failed_samples": spearman_failed,
        "spearman_failed_details": spearman_failed_details,
    }


def _save_summary(summary, output_file):
    output_file = Path(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _save_failure_report(report_dir, evaluated, failed, failed_output_file=None):
    spearman_failed_runs = []
    total_spearman_failed_samples = 0

    for run_dir, summary, output_file in evaluated:
        failed_details = summary.get("spearman_failed_details", [])
        failed_count = int(summary.get("spearman_failed_samples", 0))
        if failed_count <= 0 and not failed_details:
            continue

        total_spearman_failed_samples += failed_count
        spearman_failed_runs.append(
            {
                "run_dir": str(run_dir),
                "evaluation_summary_path": str(output_file),
                "approach": summary.get("approach"),
                "spearman_failed_samples": failed_count,
                "spearman_failed_details": failed_details,
            }
        )

    report_payload = {
        "report_dir": str(report_dir),
        "failed_run_count": len(failed),
        "failed_runs": [
            {"run_dir": run_path, "error": err}
            for run_path, err in failed
        ],
        "spearman_failed_run_count": len(spearman_failed_runs),
        "spearman_failed_sample_count": total_spearman_failed_samples,
        "spearman_failed_runs": spearman_failed_runs,
    }

    default_output = Path(report_dir) / "evaluation_failures.json"
    output_path = Path(failed_output_file) if failed_output_file else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)

    return output_path, report_payload


def _print_summary(summary, output_file):
    print("=== Evaluation summary ===")
    print(f"report_path: {summary['report_path']}")
    print(f"model: {summary['model']}")
    print(f"approach: {summary['approach']}")
    print(f"num_ok_samples_evaluated: {summary['num_ok_samples_evaluated']}")
    print(f"attack_success_rate: {summary['attack_success_rate']:.6f}")
    print(f"mean_margin_loss: {summary['mean_margin_loss']:.6f}")
    print(f"mean_saliency_loss: {summary['mean_saliency_loss']:.6f}")
    print(
        "mean_spearman_adv_vs_clean_saliency: "
        f"{summary['mean_spearman_adv_vs_clean_saliency']:.6f}"
    )
    print(f"spearman_failed_samples: {summary['spearman_failed_samples']}")
    if summary.get("spearman_failed_details"):
        print("spearman_failed_files:")
        for item in summary["spearman_failed_details"]:
            reason = item.get("reason", "unknown")
            clean_map = item.get("clean_map_path", "")
            adv_map = item.get("adv_map_path", "")
            print(f"- reason={reason}")
            print(f"  clean_map={clean_map}")
            print(f"  adv_map={adv_map}")
    print(f"saved_to: {output_file}")


def _evaluate_all_runs_in_report_dir(report_dir, max_samples=None):
    report_dir = Path(report_dir)
    if not report_dir.exists() or not report_dir.is_dir():
        raise FileNotFoundError(f"report directory not found: {report_dir}")

    run_dirs = [d for d in sorted(report_dir.iterdir()) if d.is_dir()]
    evaluated = []
    failed = []

    for run_dir in run_dirs:
        report_path = run_dir / "batch_report.json"
        if not report_path.exists():
            failed.append((str(run_dir), "missing batch_report.json"))
            continue

        try:
            summary = _evaluate_single_report(report_path=report_path, max_samples=max_samples)
        except Exception as exc:
            failed.append((str(run_dir), str(exc)))
            continue

        output_file = run_dir / "evaluation_summary.json"
        _save_summary(summary, output_file)
        evaluated.append((run_dir, summary, output_file))

    return evaluated, failed


def main():
    args = parse_args()

    if args.report_dir:
        include_models = _parse_csv_filter(args.include_models)
        include_algorithms = _parse_csv_filter(args.include_algorithms)

        evaluated, failed = _evaluate_all_runs_in_report_dir(
            report_dir=args.report_dir,
            max_samples=args.max_samples,
        )

        if not evaluated:
            raise ValueError(
                "No batch_report.json found (or all evaluations failed) under --report-dir"
            )

        print("=== Batch directory evaluation ===")
        print(f"report_dir: {args.report_dir}")
        print(f"evaluated_runs: {len(evaluated)}")
        print(f"failed_or_skipped_runs: {len(failed)}")

        for idx, (run_dir, summary, output_file) in enumerate(evaluated, start=1):
            print(
                f"[{idx}] saved: {output_file} "
                f"(ASR={summary['attack_success_rate']:.4f}, "
                f"Spearman={summary['mean_spearman_adv_vs_clean_saliency']:.4f})"
            )

        if failed:
            print("=== Skipped/failed runs ===")
            for run_path, err in failed:
                print(f"- {run_path}: {err}")

        failed_report_path, failed_report = _save_failure_report(
            report_dir=args.report_dir,
            evaluated=evaluated,
            failed=failed,
            failed_output_file=args.failed_output_file,
        )

        if (
            failed_report["failed_run_count"] > 0
            or failed_report["spearman_failed_sample_count"] > 0
        ):
            print("=== Failure report ===")
            print(f"failed_runs: {failed_report['failed_run_count']}")
            print(
                "spearman_failed_samples: "
                f"{failed_report['spearman_failed_sample_count']}"
            )
            print(f"saved_failure_report_to: {failed_report_path}")

        if args.print_latex_lines:
            _print_latex_rows_from_report_dir(args.report_dir)
        if args.print_markdown_tables:
            markdown_text = _print_markdown_tables_from_report_dir(args.report_dir)
            print(markdown_text)

            if args.markdown_output_file:
                markdown_output_path = Path(args.markdown_output_file)
                markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(markdown_output_path, "w", encoding="utf-8") as f:
                    f.write(markdown_text + "\n")
                print(f"saved_markdown_table_to: {markdown_output_path}")

        if args.filled_output_file:
            filled_output_path, filled_rows = _save_filled_output(
                report_dir=args.report_dir,
                evaluated=evaluated,
                filled_output_file=args.filled_output_file,
                include_models=include_models,
                include_algorithms=include_algorithms,
                output_format=args.filled_output_format,
            )
            print(f"saved_filled_output_to: {filled_output_path}")
            print(f"filled_output_format: {args.filled_output_format}")
            print(f"filled_rows: {filled_rows}")
        return

    if not args.report_path:
        raise ValueError("Please provide --report-path for single evaluation")

    report_path = Path(args.report_path)
    summary = _evaluate_single_report(
        report_path=report_path,
        max_samples=args.max_samples,
    )

    output_file = Path(args.output_file) if args.output_file else report_path.parent / "evaluation_summary.json"
    _save_summary(summary, output_file)
    _print_summary(summary, output_file)


if __name__ == "__main__":
    main()
