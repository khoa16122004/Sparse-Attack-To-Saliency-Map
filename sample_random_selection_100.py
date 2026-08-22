import argparse
import json
from pathlib import Path
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly sample K items from each *_selection.json file per model"
    )
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=Path("model_evaluation_results"),
        help="Directory that contains *_selection.json files",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of samples to keep for each model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional global seed for reproducible sampling",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_selection_random100.json",
        help="Suffix for generated sampled files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist",
    )
    return parser.parse_args()


def is_model_selection_file(path: Path) -> bool:
    name = path.name
    return name.endswith("_selection.json") and "_random" not in name


def load_selection(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Selection file must be a JSON object: {path}")
    return data


def make_output_path(input_path: Path, output_suffix: str) -> Path:
    model_name = input_path.name[: -len("_selection.json")]
    return input_path.parent / f"{model_name}{output_suffix}"


def sample_selection_items(data: dict, k: int, rng: random.Random) -> dict:
    items = list(data.items())
    if len(items) < k:
        raise ValueError(f"Only {len(items)} items, cannot sample k={k}")
    sampled = rng.sample(items, k)
    return {key: value for key, value in sampled}


def main():
    args = parse_args()

    if args.k <= 0:
        raise ValueError("--k must be > 0")

    selection_dir = args.selection_dir
    if not selection_dir.exists() or not selection_dir.is_dir():
        raise FileNotFoundError(f"Selection directory not found: {selection_dir}")

    selection_files = sorted(
        path for path in selection_dir.glob("*_selection.json") if is_model_selection_file(path)
    )

    if not selection_files:
        raise FileNotFoundError(f"No model selection files found in: {selection_dir}")

    global_rng = random.Random(args.seed) if args.seed is not None else random.Random()

    print(f"[INFO] selection_dir={selection_dir}")
    print(f"[INFO] models={len(selection_files)}")
    print(f"[INFO] k={args.k}")
    print(f"[INFO] seed={args.seed}")

    for selection_file in selection_files:
        model_name = selection_file.name[: -len("_selection.json")]
        output_path = make_output_path(selection_file, args.output_suffix)

        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] {model_name}: output exists -> {output_path.name}")
            continue

        data = load_selection(selection_file)

        model_seed = global_rng.randint(0, 2**31 - 1)
        model_rng = random.Random(model_seed)
        sampled = sample_selection_items(data, args.k, model_rng)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sampled, f, indent=2, ensure_ascii=False)

        print(
            f"[OK] {model_name}: {len(data)} -> {len(sampled)} | "
            f"seed={model_seed} | {output_path.name}"
        )


if __name__ == "__main__":
    main()
