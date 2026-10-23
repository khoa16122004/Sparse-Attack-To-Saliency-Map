"""Audit the expected experiment grid without computing scores or loading images."""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__:
    from .run_score import _normalize_algorithm, _parse_approach, _resolve_run_context
else:
    from run_score import _normalize_algorithm, _parse_approach, _resolve_run_context


@dataclass(frozen=True, order=True)
class Combination:
    seed: str
    model: str
    explain_method: str
    algorithm: str
    w_m: float
    w_s: float
    eps: int
    loss_type: str


@dataclass
class InventoryRun:
    combination: Combination
    run_dir: str
    samples_total: Optional[int]
    samples_ok: Optional[int]
    errors: List[str]


LOSSES = ("margin_loss", "negative_cross_entropy_saliency")
METHODS = (
    ("weighted_sum_ga", 0.0, 1.0),
    ("weighted_sum_ga", 0.5, 0.5),
    ("weighted_sum_ga", 1.0, 0.0),
    ("nsgaii", 0.5, 0.5),
)


def _combination(payload: dict) -> Combination:
    wm, ws = payload.get("w_m"), payload.get("w_s")
    if wm is None or ws is None:
        raise ValueError("Missing objective weights w_m/w_s")
    return Combination(
        seed=str(payload["seed"]),
        model=str(payload["model"]).strip().lower(),
        explain_method=str(payload["explain_method"]),
        algorithm=_normalize_algorithm(str(payload["algorithm"])),
        w_m=round(float(wm), 9),
        w_s=round(float(ws), 9),
        eps=int(payload["eps"]),
        loss_type=str(payload["loss_type"]),
    )


def scan_root(root: Path, check_samples: bool = False) -> Tuple[List[InventoryRun], List[str]]:
    if not root.is_dir():
        raise NotADirectoryError(f"Result root not found: {root}")
    runs: List[InventoryRun] = []
    issues: List[str] = []

    def walk_error(error: OSError) -> None:
        issues.append(str(error))

    for directory, children, _ in os.walk(root, onerror=walk_error):
        run_dir = Path(directory)
        if not run_dir.name.startswith("strategy-"):
            continue
        children.clear()
        meta = _parse_approach(run_dir.name)
        if meta["strategy"] != "uniform":
            continue
        model, algorithm, seed = _resolve_run_context(run_dir, root)
        meta.update(model=model, algorithm=algorithm, seed=seed)
        try:
            combination = _combination(meta)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"{run_dir}: {exc}")
            continue

        if not check_samples:
            runs.append(InventoryRun(combination, str(run_dir), None, None, []))
            continue

        # The two names are aliases for one sample, not two samples.
        summaries = {
            path.parent: path for path in sorted(run_dir.glob("*/*/summarize.json"))
        }
        summaries.update({
            path.parent: path for path in sorted(run_dir.glob("*/*/summary.json"))
        })
        errors: List[str] = []
        ok = 0
        for path in summaries.values():
            try:
                with path.open(encoding="utf-8") as handle:
                    sample = json.load(handle)
                if not isinstance(sample, dict):
                    raise ValueError("Expected a sample JSON object")
                if sample.get("status") == "ok":
                    ok += 1
            except (OSError, ValueError) as exc:
                errors.append(f"{path}: {exc}")
        runs.append(InventoryRun(
            combination, str(run_dir), len(summaries), ok, errors
        ))
    return runs, issues


def load_cache(path: Path) -> List[InventoryRun]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected an all_runs.json list")
    runs: List[InventoryRun] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: entry {index} is not an object")
        if item.get("strategy") != "uniform":
            continue
        try:
            total = int(item["num_samples_total"])
            ok = int(item["num_samples_ok"])
            if not 0 <= ok <= total:
                raise ValueError("Expected 0 <= num_samples_ok <= num_samples_total")
            runs.append(InventoryRun(
                _combination(item), str(item["run_dir"]), total, ok, []
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid entry {index}: {exc}") from exc
    return runs


def expected_grid(args: argparse.Namespace) -> List[Combination]:
    seeds = [args.seed] if args.seed != "all" else args.expected_seeds
    models = [args.model.lower()] if args.model != "all" else args.expected_models
    explanations = (
        [args.explain_method] if args.explain_method != "all"
        else args.expected_explain_methods
    )
    epsilons = [int(args.eps)] if args.eps != "all" else [20, 50, 100]
    algorithm = _normalize_algorithm(args.algorithm)
    methods = [
        method for method in METHODS
        if args.algorithm == "all" or method[0] == algorithm
    ]
    if not methods:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")
    return sorted({
        Combination(seed, model.lower(), explanation, algo, wm, ws, eps, loss)
        for seed, model, explanation, (algo, wm, ws), eps, loss in product(
            seeds, models, explanations, methods, epsilons, LOSSES
        )
    })


def build_report(
    runs: List[InventoryRun],
    expected: List[Combination],
    expected_samples: int = 100,
    min_sample_ratio: float = 0.5,
) -> dict:
    indexed: Dict[Combination, List[InventoryRun]] = defaultdict(list)
    for run in runs:
        indexed[run.combination].append(run)
    rows = []
    groups: dict = {}
    for combination in expected:
        candidates = indexed.get(combination, [])
        best = max(
            candidates,
            key=lambda run: run.samples_ok if run.samples_ok is not None else -1,
            default=None,
        )
        samples_checked = best is not None and best.samples_ok is not None
        ok = best.samples_ok if best and best.samples_ok is not None else 0
        target = max(expected_samples, (best.samples_total or 0) if best else 0)
        status = (
            "missing" if best is None else
            "present" if not samples_checked else
            "complete" if ok >= target else
            "partial" if ok > 0 and ok / target >= min_sample_ratio else
            "insufficient"
        )
        equivalent = None
        if not candidates and combination.w_m == 0 and combination.w_s == 1:
            other_loss = LOSSES[1] if combination.loss_type == LOSSES[0] else LOSSES[0]
            other = Combination(**{**asdict(combination), "loss_type": other_loss})
            if other in indexed:
                equivalent = other_loss
        rows.append({
            **asdict(combination),
            "status": status,
            "samples_checked": samples_checked,
            "samples_ok": ok if samples_checked else None,
            "samples_total": best.samples_total if best else 0,
            "expected_samples": target,
            "samples_missing": max(0, target - ok) if samples_checked else None,
            "usable_for_score": (
                ok > 0 and ok / target >= min_sample_ratio
                if samples_checked else None
            ),
            "run_dirs": [run.run_dir for run in candidates],
            "errors": [error for run in candidates for error in run.errors],
            "equivalent_loss_available": equivalent,
        })
        group_key = (combination.seed, combination.model, combination.explain_method)
        if group_key not in groups:
            groups[group_key] = Counter()
        groups[group_key][status] += 1
    expected_set = set(expected)
    return {
        "summary": dict(Counter(row["status"] for row in rows)),
        "expected_combinations": len(expected),
        "groups": [
            {"seed": seed, "model": model, "explain_method": explanation,
             "expected": sum(counts.values()), **dict(counts)}
            for (seed, model, explanation), counts in groups.items()
        ],
        "combinations": rows,
        "unexpected_runs": [
            asdict(run) for run in runs if run.combination not in expected_set
        ],
    }


def render_text(report: dict) -> str:
    lines = [
        "EXPERIMENT COMPLETENESS REPORT",
        f"Source: {report['source']}",
        f"Expected combinations: {report['expected_combinations']}",
        "Status counts: " + json.dumps(report["summary"], sort_keys=True),
        "present=run folder exists (samples NOT checked); "
        "complete=all samples; partial=usable but unfinished; "
        "insufficient=below sample threshold; missing=no run.",
    ]
    lines.extend(f"WARNING: {warning}" for warning in report["warnings"])
    lines.append("\nBY SEED / MODEL / EXPLAIN METHOD")
    for group in report["groups"]:
        counts = " ".join(
            f"{name}={group.get(name, 0)}"
            for name in ("present", "complete", "partial", "insufficient", "missing")
        )
        lines.append(
            f"{group['seed']} | {group['model']} | {group['explain_method']} | {counts}"
        )
    lines.append("\nALL COMBINATIONS")
    for row in report["combinations"]:
        samples = (
            f"{row['samples_ok']}/{row['expected_samples']}"
            if row["samples_checked"] else "not checked"
        )
        lines.append(
            f"{row['status'].upper()} | seed={row['seed']} | {row['model']} | "
            f"{row['explain_method']} | {row['algorithm']} "
            f"weights={row['w_m']:g}:{row['w_s']:g} | eps={row['eps']} | "
            f"{row['loss_type']} | samples={samples}"
        )
        if row["equivalent_loss_available"]:
            lines.append(f"  Equivalent loss available: {row['equivalent_loss_available']}")
        lines.extend(f"  Run: {path}" for path in row["run_dirs"])
        lines.extend(f"  ERROR: {error}" for error in row["errors"])
    lines.append(f"\nUnexpected/out-of-scope runs: {len(report['unexpected_runs'])}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--result-root", dest="root", default="server_run/server_run")
    parser.add_argument("--output-dir", default="evaluate_script/stats_outputs")
    parser.add_argument("--all-runs-json")
    parser.add_argument("--cached", "--load-cached", action="store_true")
    parser.add_argument(
        "--check-samples", action="store_true",
        help="Read per-sample summary JSON to check sample completeness (slower). "
        "Default: only check run-folder presence, without visiting samples.",
    )
    parser.add_argument("--seed", default="all")
    parser.add_argument("--model", default="all")
    parser.add_argument("--algorithm", default="all")
    parser.add_argument("--explain-method", default="all")
    parser.add_argument("--eps", default="all")
    parser.add_argument("--expected-seeds", nargs="+", default=[
        str(seed) for seed in range(22520691, 22520696)
    ])
    parser.add_argument("--expected-models", nargs="+", default=["resnet18", "vgg16", "vit_b_32"])
    parser.add_argument("--expected-explain-methods", nargs="+", default=[
        "simple_gradient", "input_gradient", "grad_cam", "raw_attention"
    ])
    parser.add_argument("--expected-samples", type=int, default=100)
    parser.add_argument("--min-sample-ratio", type=float, default=0.5)
    args = parser.parse_args()
    if args.expected_samples < 1:
        parser.error("--expected-samples must be positive")
    if not 0 <= args.min_sample_ratio <= 1:
        parser.error("--min-sample-ratio must be between 0 and 1")
    if args.eps != "all":
        try:
            int(args.eps)
        except ValueError:
            parser.error("--eps must be an integer or 'all'")
    return args


def main() -> None:
    args = parse_args()
    expected = expected_grid(args)
    root = Path(args.root)
    output = Path(args.output_dir)
    cache: Optional[Path] = (
        Path(args.all_runs_json) if args.all_runs_json else
        output / "all_runs.json" if args.cached else
        root if root.suffix.lower() == ".json" else None
    )
    if cache is not None:
        runs = load_cache(cache)
        warnings = [
            "Cache-only audit: filtered-out runs cannot be distinguished from absent runs. "
            "run_score may also have mirrored equivalent losses. Scan --root for "
            "physical run completeness; cached paths are not checked on disk."
        ]
    else:
        print(
            "Scanning sample summaries..." if args.check_samples else
            "Checking run folders only (no sample files read)...",
            flush=True,
        )
        runs, warnings = scan_root(root, check_samples=args.check_samples)
        if not runs:
            warnings.append("No recognized uniform run folders found under the supplied root.")
    report = build_report(runs, expected, args.expected_samples, args.min_sample_ratio)
    report.update(source=str(cache if cache is not None else root), warnings=warnings)
    text = render_text(report)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "completeness_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    with (output / "completeness_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(text)
    with (output / "completeness_report.csv").open("w", encoding="utf-8", newline="") as handle:
        rows = [
            {key: json.dumps(value) if isinstance(value, list) else value
             for key, value in row.items()}
            for row in report["combinations"]
        ]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(text, end="")
    print(f"\nSaved completeness_report.txt / .json / .csv under: {output}")


if __name__ == "__main__":
    main()
