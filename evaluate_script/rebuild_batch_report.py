import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild batch_report.json from existing per-sample summary.json files. "
            "You can target one run directory or scan a root directory."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Single run directory (contains class/image/summary.json files)",
    )
    target.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root to scan for run directories (supports root or model/root style)",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="batch_report.json",
        help="Output report filename inside each run directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output report",
    )
    parser.add_argument(
        "--selection-file",
        type=str,
        default=None,
        help="Optional selection_file value to write into rebuilt report",
    )
    parser.add_argument(
        "--num-requested",
        type=int,
        default=None,
        help="Optional num_requested value to write into rebuilt report",
    )
    return parser.parse_args()


def _looks_like_run_dir(path: Path) -> bool:
    return any(path.glob("*/*/summary.json"))


def _discover_run_dirs(root: Path) -> List[Path]:
    run_dirs: List[Path] = []

    if _looks_like_run_dir(root):
        return [root]

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue

        if _looks_like_run_dir(child):
            run_dirs.append(child)
            continue

        for sub in sorted(child.iterdir()):
            if not sub.is_dir():
                continue
            if _looks_like_run_dir(sub):
                run_dirs.append(sub)

    # De-duplicate while preserving order.
    unique: List[Path] = []
    seen = set()
    for run_dir in run_dirs:
        key = str(run_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(run_dir)
    return unique


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_dict(value) -> Dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def _infer_model_and_approach(run_dir: Path) -> Dict[str, str]:
    inferred = {"model": "unknown", "approach": run_dir.name}
    parts = run_dir.parts
    if len(parts) >= 2:
        inferred["model"] = parts[-2]
    return inferred


def _status_counts(results: List[Dict[str, object]]) -> Dict[str, int]:
    counts = {
        "ok": 0,
        "failed": 0,
        "missing_image": 0,
        "skipped": 0,
    }

    for item in results:
        status = str(item.get("status", "failed"))
        if status == "ok":
            counts["ok"] += 1
        elif status == "missing_image":
            counts["missing_image"] += 1
        elif status == "skipped":
            counts["skipped"] += 1
        else:
            counts["failed"] += 1

    return counts


def _read_summaries(run_dir: Path) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for summary_path in sorted(run_dir.glob("*/*/summary.json")):
        payload = _safe_dict(_load_json(summary_path))
        if "output_dir" not in payload:
            payload["output_dir"] = str(summary_path.parent)
        results.append(payload)
    return results


def _build_report(
    run_dir: Path,
    results: List[Dict[str, object]],
    existing_report: Dict[str, object],
    selection_file_override: Optional[str],
    num_requested_override: Optional[int],
) -> Dict[str, object]:
    inferred = _infer_model_and_approach(run_dir)

    model = "unknown"
    if results:
        model = str(results[0].get("model") or "") or inferred["model"]
    if model == "unknown":
        model = str(existing_report.get("model") or inferred["model"])

    approach = str(existing_report.get("approach") or inferred["approach"])

    if selection_file_override is not None:
        selection_file = selection_file_override
    else:
        selection_file = str(existing_report.get("selection_file") or "")

    if num_requested_override is not None:
        num_requested = num_requested_override
    else:
        num_requested = existing_report.get("num_requested")

    counts = _status_counts(results)

    return {
        "selection_file": selection_file,
        "model": model,
        "approach": approach,
        "num_requested": num_requested,
        "total": len(results),
        "ok": counts["ok"],
        "failed": counts["failed"],
        "missing_image": counts["missing_image"],
        "skipped": counts["skipped"],
        "imagenet_val_root": str(existing_report.get("imagenet_val_root") or ""),
        "replace_from_root": str(existing_report.get("replace_from_root") or ""),
        "results": results,
    }


def _rebuild_one_run(
    run_dir: Path,
    output_name: str,
    overwrite: bool,
    selection_file_override: Optional[str],
    num_requested_override: Optional[int],
) -> Optional[Path]:
    output_path = run_dir / output_name
    if output_path.exists() and not overwrite:
        print(f"[SKIP] exists: {output_path}")
        return None

    results = _read_summaries(run_dir)
    if not results:
        print(f"[SKIP] no summary.json found: {run_dir}")
        return None

    existing_report = {}
    if output_path.exists():
        try:
            existing_report = _safe_dict(_load_json(output_path))
        except Exception:
            existing_report = {}

    report = _build_report(
        run_dir=run_dir,
        results=results,
        existing_report=existing_report,
        selection_file_override=selection_file_override,
        num_requested_override=num_requested_override,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(
        f"[OK] {output_path} | total={report['total']} ok={report['ok']} "
        f"failed={report['failed']} missing={report['missing_image']} skipped={report['skipped']}"
    )
    return output_path


def main() -> None:
    args = parse_args()

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        root = Path(args.root)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"root directory not found: {root}")
        run_dirs = _discover_run_dirs(root)

    if not run_dirs:
        raise ValueError("No run directory with summary.json found")

    written = 0
    skipped = 0

    for run_dir in run_dirs:
        if not run_dir.exists() or not run_dir.is_dir():
            print(f"[SKIP] invalid directory: {run_dir}")
            skipped += 1
            continue

        out = _rebuild_one_run(
            run_dir=run_dir,
            output_name=args.output_name,
            overwrite=args.overwrite,
            selection_file_override=args.selection_file,
            num_requested_override=args.num_requested,
        )
        if out is None:
            skipped += 1
        else:
            written += 1

    print("=== Rebuild summary ===")
    print(f"run_dirs: {len(run_dirs)}")
    print(f"written: {written}")
    print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
