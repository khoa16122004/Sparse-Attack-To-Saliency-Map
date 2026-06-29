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
            "Margin L-oss, Saliency Loss, and Spearman rank correlation"
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
            "Path to directory containing multiple run folders with "
            "evaluation_summary.json"
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
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only first N successful samples",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Where to save evaluation JSON. Default: <run_root>/evaluation_summary.json",
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


def _format_asr_percent(asr):
    return f"{(float(asr) * 100.0):.2f}"


def _format_spearman(score):
    return f"{float(score):.4f}"


        if args.report_dir:
            success, failures = _evaluate_all_runs_in_report_dir(
                report_dir=args.report_dir,
                max_samples=args.max_samples,
            )

            print("=== Batch directory evaluation ===")
            print(f"report_dir: {args.report_dir}")
            print(f"evaluated_runs: {len(success)}")
            print(f"failed_or_skipped_runs: {len(failures)}")

            for run_name, output_file, _ in success:
                print(f"[OK] {run_name} -> {output_file}")

            for run_name, reason in failures:
                print(f"[SKIP] {run_name} -> {reason}")

            if args.print_latex_lines:
                _print_latex_rows_from_report_dir(args.report_dir)

            return
    return text


        summary, output_file = _evaluate_report_file(
            report_path=args.report_path,
            max_samples=args.max_samples,
            output_file=args.output_file,
        )

        print(
            f"& {eps_value} & {margin_asr_txt} & {margin_spearman_txt} "
            f"& {ce_asr_txt} & {ce_spearman_txt} \\\\"
        )


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


def _evaluate_report_file(report_path, max_samples=None, output_file=None):
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

    for result in tqdm(ok_results, desc=f"Evaluating {report_path.parent.name}", unit="img"):
        true_label = int(result.get("true_label", result.get("clean_pred", -1)))
        adv_pred = int(result.get("adv_pred", -1))

        margin_losses.append(float(result.get("margin_loss", float("nan"))))
        saliency_losses.append(float(result.get("saliency_loss", float("nan"))))

        success_flags.append(int(adv_pred != true_label and true_label >= 0 and adv_pred >= 0))

        corr, err = _compute_spearman_for_sample(
            result=result,
        )
        if err is not None:
            spearman_failed += 1
        spearman_scores.append(corr)

    attack_success_rate = float(np.mean(success_flags))
    mean_margin_loss = _safe_mean(margin_losses)
    mean_saliency_loss = _safe_mean(saliency_losses)
    mean_spearman = _safe_mean(spearman_scores)

    summary = {
        "report_path": str(report_path),
        "model": model_name,
        "approach": report.get("approach"),
        "num_ok_samples_evaluated": len(ok_results),
        "attack_success_rate": attack_success_rate,
        "mean_margin_loss": mean_margin_loss,
        "mean_saliency_loss": mean_saliency_loss,
        "mean_spearman_adv_vs_clean_saliency": mean_spearman,
        "spearman_failed_samples": spearman_failed,
    }

    output_file = Path(output_file) if output_file else report_path.parent / "evaluation_summary.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary, output_file


def _evaluate_all_runs_in_report_dir(report_dir, max_samples=None):
    root = Path(report_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"report directory not found: {root}")

    run_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if not run_dirs:
        raise ValueError(f"No run directories found in: {root}")

    success = []
    failures = []

    for run_dir in run_dirs:
        report_path = run_dir / "batch_report.json"
        if not report_path.exists():
            failures.append((run_dir.name, "missing batch_report.json"))
            continue

        try:
            summary, output_file = _evaluate_report_file(
                report_path=report_path,
                max_samples=max_samples,
                output_file=run_dir / "evaluation_summary.json",
            )
            success.append((run_dir.name, output_file, summary))
        except Exception as e:
            failures.append((run_dir.name, str(e)))

    return success, failures


def _compute_spearman_for_sample(result):
    output_dir = Path(result["output_dir"])
    clean_map_path = output_dir / "clean_map.png"
    adv_map_path = output_dir / "adv_map.png"

    if not clean_map_path.exists() or not adv_map_path.exists():
        return None, "missing_clean_or_adv_map"

    clean_map = _load_gray_image_array(clean_map_path)
    adv_map = _load_gray_image_array(adv_map_path)

    corr = _spearman_rank_corr(clean_map, adv_map)
    return corr, None


def _evaluate_single_report(report_path, max_samples=None):
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

    for result in tqdm(ok_results, desc=f"Evaluating {report_path.parent.name}", unit="img"):
        true_label = int(result.get("true_label", result.get("clean_pred", -1)))
        adv_pred = int(result.get("adv_pred", -1))

        margin_losses.append(float(result.get("margin_loss", float("nan"))))
        saliency_losses.append(float(result.get("saliency_loss", float("nan"))))

        success_flags.append(int(adv_pred != true_label and true_label >= 0 and adv_pred >= 0))

        corr, err = _compute_spearman_for_sample(
            result=result,
        )
        if err is not None:
            spearman_failed += 1
        spearman_scores.append(corr)

    attack_success_rate = float(np.mean(success_flags))
    mean_margin_loss = _safe_mean(margin_losses)
    mean_saliency_loss = _safe_mean(saliency_losses)
    mean_spearman = _safe_mean(spearman_scores)

    summary = {
        "report_path": str(report_path),
        "model": model_name,
        "approach": report.get("approach"),
        "num_ok_samples_evaluated": len(ok_results),
        "attack_success_rate": attack_success_rate,
        "mean_margin_loss": mean_margin_loss,
        "mean_saliency_loss": mean_saliency_loss,
        "mean_spearman_adv_vs_clean_saliency": mean_spearman,
        "spearman_failed_samples": spearman_failed,
    }
    return summary


def _save_summary(summary, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


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
    print(f"saved_to: {output_file}")


def main():
    args = parse_args()

    if args.report_dir:
        report_dir = Path(args.report_dir)
        if not report_dir.exists() or not report_dir.is_dir():
            raise FileNotFoundError(f"report directory not found: {report_dir}")

        run_dirs = [d for d in sorted(report_dir.iterdir()) if d.is_dir()]
        evaluated = 0
        failed = []

        for run_dir in run_dirs:
            report_path = run_dir / "batch_report.json"
            if not report_path.exists():
                continue

            try:
                summary = _evaluate_single_report(
                    report_path=report_path,
                    max_samples=args.max_samples,
                )
            except Exception as exc:
                failed.append((str(run_dir), str(exc)))
                continue

            output_file = run_dir / "evaluation_summary.json"
            _save_summary(summary, output_file)
            print(
                f"[{evaluated + 1}] saved: {output_file} "
                f"(ASR={summary['attack_success_rate']:.4f}, "
                f"Spearman={summary['mean_spearman_adv_vs_clean_saliency']:.4f})"
            )
            evaluated += 1

        if evaluated == 0:
            raise ValueError(
                "No batch_report.json found (or all evaluations failed) under --report-dir"
            )

        if failed:
            print("=== Skipped/failed runs ===")
            for run_path, err in failed:
                print(f"- {run_path}: {err}")

        print(f"Evaluated {evaluated} run(s) under: {report_dir}")

        if args.print_latex_lines:
            _print_latex_rows_from_report_dir(args.report_dir)
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
