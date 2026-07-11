import argparse
from pathlib import Path

import numpy as np


def parse_points(file_path: Path) -> np.ndarray:
    points = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # Support either whitespace-separated or comma-separated values.
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) < 2:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric values at line {line_no} in {file_path}: {raw_line.rstrip()}"
                ) from exc

            points.append((x, y))

    if not points:
        raise ValueError(f"No valid 2D points found in {file_path}")

    return np.asarray(points, dtype=np.float64)


def build_output_path(input_path: Path, output_path: str | None) -> Path:
    if output_path:
        return Path(output_path)
    return input_path.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a 2D non-dominated front from a text file (each line: score1 score2)."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to non_dominated_front_scores.txt")
    parser.add_argument("--output", type=str, default=None, help="Output image path (default: same name with .png)")
    parser.add_argument("--title", type=str, default=None, help="Custom figure title")
    parser.add_argument("--xlabel", type=str, default="Score 1", help="X-axis label")
    parser.add_argument("--ylabel", type=str, default="Score 2", help="Y-axis label")
    parser.add_argument("--dpi", type=int, default=180, help="Saved image DPI")
    parser.add_argument("--show", action="store_true", help="Display plot window")
    parser.add_argument(
        "--no-connect",
        action="store_true",
        help="Do not connect points in x-sorted order",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = build_output_path(input_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points = parse_points(input_path)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.scatter(points[:, 0], points[:, 1], s=30, alpha=0.9, label="Front points")

    if not args.no_connect and points.shape[0] >= 2:
        sort_idx = np.argsort(points[:, 0])
        sorted_points = points[sort_idx]
        ax.plot(sorted_points[:, 0], sorted_points[:, 1], linewidth=1.2, alpha=0.8)

    title = args.title if args.title else f"Non-dominated front: {input_path.name}"
    ax.set_title(title)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)

    print(f"Saved plot: {output_path}")
    print(f"Total points: {points.shape[0]}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
