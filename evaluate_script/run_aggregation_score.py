import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


def _safe_name(path: Path) -> str:
    raw = path.name.strip() or "run"
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in raw)


def _resolve_plot_script(custom_script: Optional[str]) -> Path:
    if custom_script:
        script_path = Path(custom_script)
    else:
        script_path = Path(__file__).resolve().parent / "plot_sample_del_ins_curves.py"
    if not script_path.exists():
        raise FileNotFoundError(f"plot script not found: {script_path}")
    return script_path.resolve()


def _split_input_dirs(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in values:
        # Support both whitespace-separated and comma-separated forms.
        for token in str(raw).split(","):
            cleaned = token.strip().strip('"').strip("'")
            if cleaned:
                out.append(cleaned)
    return out


def _has_summary_files(folder: Path) -> bool:
    return any(folder.glob("**/summary.json")) or any(folder.glob("**/summarize.json"))


def _discover_run_targets(root: Path) -> List[Path]:
    """Discover aggregate targets under one root.

    Preference order:
    1) Direct child folders that contain summary files somewhere below (common pk split).
    2) Fallback to root itself if it contains summary files.
    """
    child_targets: List[Path] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if _has_summary_files(child):
            child_targets.append(child)

    if child_targets:
        return child_targets
    if _has_summary_files(root):
        return [root]
    return []


def _build_output_key(root: Path, run_dir: Path) -> str:
    root_key = _safe_name(root)
    try:
        rel = run_dir.relative_to(root)
    except ValueError:
        return f"{root_key}__{_safe_name(run_dir)}"

    rel_tokens = [_safe_name(Path(part)) for part in rel.parts if part.strip()]
    if not rel_tokens:
        return root_key
    return f"{root_key}__{'__'.join(rel_tokens)}"


def _build_command(
    python_exec: str,
    plot_script: Path,
    run_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        python_exec,
        str(plot_script),
        "--run-dir",
        str(run_dir),
        "--output-root",
        str(output_root),
        "--skip-sample-plots",
        "--dpi",
        str(args.dpi),
        "--aggregate-y-min",
        str(args.aggregate_y_min),
        "--aggregate-y-max",
        str(args.aggregate_y_max),
        "--aggregate-x-min",
        str(args.aggregate_x_min),
        "--aggregate-x-max",
        str(args.aggregate_x_max),
        "--aggregate-marker-count",
        str(args.aggregate_marker_count),
        "--aggregate-iteration-marker-count",
        str(args.aggregate_iteration_marker_count),
        "--aggregate-iteration-start",
        str(args.aggregate_iteration_start),
        "--aggregate-iteration-end",
        str(args.aggregate_iteration_end),
        "--objective-y-pad-ratio",
        str(args.objective_y_pad_ratio),
    ]

    if args.aggregate_auto_y:
        cmd.append("--aggregate-auto-y")
        cmd.extend(["--aggregate-auto-y-pad", str(args.aggregate_auto_y_pad)])

    if args.allow_non_ok:
        cmd.append("--allow-non-ok")
    if args.aggregate_points_only:
        cmd.append("--aggregate-points-only")
    if args.skip_existing:
        cmd.append("--skip-existing")

    if args.run_model:
        cmd.extend(["--run-model", args.run_model])
    if args.run_explain:
        cmd.extend(["--run-explain", args.run_explain])
    if args.run_approach:
        cmd.extend(["--run-approach", args.run_approach])

    if args.objective_y_min is not None and args.objective_y_max is not None:
        cmd.extend(["--objective-y-min", str(args.objective_y_min)])
        cmd.extend(["--objective-y-max", str(args.objective_y_max)])

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch aggregate runner for plot_sample_del_ins_curves.py. "
            "Inputs one or more run folders and writes outputs under aggration_score/. "
            "Objective triptych uses history saliency as Jacard Sim; Spearman is kept for final bar summaries."
        )
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help=(
            "One or more input roots. Accepts whitespace-separated values or comma-separated form "
            "(example: --input-dirs D:/RIVF_result, D:/other_result)."
        ),
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="aggration_score",
        help="Base output directory. Each input folder gets its own subfolder here.",
    )
    parser.add_argument(
        "--plot-script",
        type=str,
        default=None,
        help="Optional explicit path to plot_sample_del_ins_curves.py.",
    )
    parser.add_argument("--run-model", type=str, default=None)
    parser.add_argument("--run-explain", type=str, default=None)
    parser.add_argument("--run-approach", type=str, default=None)
    parser.add_argument("--allow-non-ok", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--dpi", type=int, default=420)
    parser.add_argument("--aggregate-y-min", type=float, default=-0.02)
    parser.add_argument("--aggregate-y-max", type=float, default=1.02)
    parser.add_argument("--aggregate-x-min", type=float, default=0.0)
    parser.add_argument("--aggregate-x-max", type=float, default=1.0)
    parser.add_argument("--aggregate-marker-count", type=int, default=100)
    parser.add_argument("--aggregate-iteration-marker-count", type=int, default=20)
    parser.add_argument("--aggregate-iteration-start", type=int, default=0)
    parser.add_argument("--aggregate-iteration-end", type=int, default=-1)
    parser.add_argument("--objective-y-pad-ratio", type=float, default=0.01)
    parser.add_argument("--objective-y-min", type=float, default=None)
    parser.add_argument("--objective-y-max", type=float, default=None)
    parser.add_argument("--aggregate-points-only", action="store_true")
    parser.add_argument("--aggregate-auto-y", action="store_true", help="Enable tight auto y-scaling in aggregate plots.")
    parser.add_argument("--no-aggregate-auto-y", action="store_false", dest="aggregate_auto_y", help="Disable tight auto y-scaling.")
    parser.add_argument("--aggregate-auto-y-pad", type=float, default=0.035)
    parser.set_defaults(aggregate_auto_y=True)
    args = parser.parse_args()

    plot_script = _resolve_plot_script(args.plot_script)
    output_base = Path(args.output_base).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    input_dir_tokens = _split_input_dirs(args.input_dirs)
    if not input_dir_tokens:
        raise ValueError("No valid input directories provided in --input-dirs")

    python_exec = sys.executable
    results: List[Dict[str, object]] = []
    used_names: Dict[str, int] = {}
    discovered_jobs: List[Tuple[Path, Path]] = []

    for raw_dir in input_dir_tokens:
        run_dir = Path(raw_dir).resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            results.append(
                {
                    "input_dir": str(run_dir),
                    "status": "failed",
                    "reason": "run directory not found or not a directory",
                }
            )
            print(f"[FAILED] invalid input dir: {run_dir}")
            continue

        targets = _discover_run_targets(run_dir)
        if not targets:
            results.append(
                {
                    "input_dir": str(run_dir),
                    "status": "failed",
                    "reason": "no summary.json/summarize.json found under this input root",
                }
            )
            print(f"[FAILED] no summaries found under: {run_dir}")
            continue

        print(f"[DISCOVER] root={run_dir} -> {len(targets)} target(s)")
        for target in targets:
            discovered_jobs.append((run_dir, target))

    # Keep (input_root, run_dir) unique while preserving order.
    unique_jobs: List[Tuple[Path, Path]] = []
    seen_job_keys: Set[str] = set()
    for input_root, run_dir in discovered_jobs:
        key = f"{str(input_root.resolve()).lower()}::{str(run_dir.resolve()).lower()}"
        if key in seen_job_keys:
            continue
        seen_job_keys.add(key)
        unique_jobs.append((input_root, run_dir))

    for input_root, run_dir in unique_jobs:
        folder_key = _build_output_key(input_root, run_dir)
        count = used_names.get(folder_key, 0)
        used_names[folder_key] = count + 1
        if count > 0:
            folder_key = f"{folder_key}_{count + 1}"

        output_root = output_base / folder_key
        output_root.mkdir(parents=True, exist_ok=True)

        cmd = _build_command(
            python_exec=python_exec,
            plot_script=plot_script,
            run_dir=run_dir,
            output_root=output_root,
            args=args,
        )

        print(f"[RUN] input_root={input_root}")
        print(f"      run_dir={run_dir}")
        print(f"      output={output_root}")
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)

        results.append(
            {
                "input_root": str(input_root),
                "input_dir": str(run_dir),
                "output_dir": str(output_root),
                "status": "ok" if proc.returncode == 0 else "failed",
                "return_code": int(proc.returncode),
                "command": cmd,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )

        if proc.returncode == 0:
            print(f"[OK] aggregation finished for: {run_dir}")
        else:
            print(f"[FAILED] aggregation failed for: {run_dir} (code={proc.returncode})")
            if proc.stderr:
                print(proc.stderr)

    summary_path = output_base / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    ok_count = sum(1 for item in results if item.get("status") == "ok")
    failed_count = sum(1 for item in results if item.get("status") != "ok")

    print("=== Done ===")
    print(f"output base: {output_base}")
    print(f"ok: {ok_count}")
    print(f"failed: {failed_count}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
