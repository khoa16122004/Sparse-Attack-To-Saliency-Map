import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torchvision.models as tv_models
from torchvision.models import get_model_weights


def _download_and_save_one(model_name: str, checkpoint_dir: Path, overwrite: bool) -> Dict[str, str]:
    output_path = checkpoint_dir / f"{model_name}.pth"
    if output_path.exists() and not overwrite:
        return {
            "model": model_name,
            "status": "skipped",
            "checkpoint": str(output_path),
            "reason": "exists",
        }

    model_fn = getattr(tv_models, model_name)
    weights_enum = get_model_weights(model_name).DEFAULT
    model = model_fn(weights=weights_enum)

    state_dict = model.state_dict()
    torch.save(state_dict, output_path)

    return {
        "model": model_name,
        "status": "ok",
        "checkpoint": str(output_path),
        "weights": str(weights_enum),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download torchvision pretrained weights and store them as local checkpoints "
            "for offline/local loading."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Torchvision model names, e.g. resnet18 resnet50 vgg16 vit_b_16",
    )
    parser.add_argument(
        "--selection-dir",
        type=str,
        default=None,
        help="Optional folder of *_selection.json files to auto-discover model names.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to store checkpoint files (*.pth).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing checkpoint files.",
    )
    return parser.parse_args()


def _collect_models(args: argparse.Namespace) -> List[str]:
    models: List[str] = []
    if args.models:
        models.extend(args.models)

    if args.selection_dir:
        selection_dir = Path(args.selection_dir)
        if selection_dir.exists() and selection_dir.is_dir():
            for p in sorted(selection_dir.glob("*_selection*.json")):
                name = p.stem
                if name.endswith("_selection_random50"):
                    name = name[: -len("_selection_random50")]
                elif name.endswith("_selection"):
                    name = name[: -len("_selection")]
                if name:
                    models.append(name)

    ordered_unique: List[str] = []
    seen = set()
    for name in models:
        if name in seen:
            continue
        seen.add(name)
        ordered_unique.append(name)

    return ordered_unique


def main() -> None:
    args = parse_args()
    model_names = _collect_models(args)
    if not model_names:
        raise ValueError("No models provided. Use --models and/or --selection-dir.")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, str]] = []
    for model_name in model_names:
        try:
            result = _download_and_save_one(model_name, checkpoint_dir, args.overwrite)
        except Exception as exc:
            result = {
                "model": model_name,
                "status": "failed",
                "error": str(exc),
            }
        results.append(result)

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "models": model_names,
        "total": len(results),
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "results": results,
    }

    summary_path = checkpoint_dir / "download_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] checkpoint_dir={checkpoint_dir}")
    for row in results:
        status = row.get("status")
        model_name = row.get("model")
        if status == "ok":
            print(f"[OK] {model_name} -> {row.get('checkpoint')}")
        elif status == "skipped":
            print(f"[SKIP] {model_name} -> {row.get('checkpoint')} ({row.get('reason')})")
        else:
            print(f"[FAILED] {model_name} error={row.get('error')}")

    print(f"[INFO] summary={summary_path}")
    print(f"[INFO] total={summary['total']} ok={summary['ok']} skipped={summary['skipped']} failed={summary['failed']}")


if __name__ == "__main__":
    main()
