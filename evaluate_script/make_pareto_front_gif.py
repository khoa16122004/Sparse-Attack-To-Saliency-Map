import argparse
import re
from pathlib import Path
from typing import List

import numpy as np


def _extract_iter_index(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def _read_front_points(path: Path) -> np.ndarray:
    points = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) < 2:
                continue
            points.append((float(parts[0]), float(parts[1])))

    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _collect_files(history_dir: Path) -> List[Path]:
    files = sorted(history_dir.glob("*.txt"), key=_extract_iter_index)
    if not files:
        raise ValueError(f"No txt files found in history dir: {history_dir}")
    return files


def _build_output_path(history_dir: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return history_dir / "pareto_front_evolution.gif"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an animated GIF showing Pareto front evolution from per-iteration txt files."
    )
    parser.add_argument("--history-dir", type=str, required=True, help="Directory with iter_XXXX.txt files")
    parser.add_argument("--output", type=str, default=None, help="Output gif path")
    parser.add_argument("--fps", type=int, default=4, help="GIF frames per second")
    parser.add_argument("--dpi", type=int, default=150, help="GIF dpi")
    parser.add_argument("--xlabel", type=str, default="Score 1")
    parser.add_argument("--ylabel", type=str, default="Score 2")
    parser.add_argument("--title-prefix", type=str, default="Pareto Front Evolution")
    parser.add_argument("--no-connect", action="store_true", help="Disable connecting points in x-sorted order")
    args = parser.parse_args()

    history_dir = Path(args.history_dir)
    if not history_dir.exists() or not history_dir.is_dir():
        raise FileNotFoundError(f"History dir not found: {history_dir}")

    files = _collect_files(history_dir)
    output_path = _build_output_path(history_dir, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fronts = [_read_front_points(p) for p in files]
    valid_fronts = [f for f in fronts if f.shape[0] > 0]
    if not valid_fronts:
        raise ValueError(f"All files in {history_dir} are empty or invalid")

    all_points = np.vstack(valid_fronts)
    x_min, y_min = np.min(all_points[:, 0]), np.min(all_points[:, 1])
    x_max, y_max = np.max(all_points[:, 0]), np.max(all_points[:, 1])

    # Add a little padding so points are not glued to border.
    x_pad = max((x_max - x_min) * 0.05, 1e-6)
    y_pad = max((y_max - y_min) * 0.05, 1e-6)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    def update(frame_idx: int):
        ax.clear()
        front = fronts[frame_idx]

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(args.ylabel)
        ax.grid(alpha=0.25)

        if front.shape[0] > 0:
            ax.scatter(front[:, 0], front[:, 1], s=26, alpha=0.9, color="#1f77b4")
            if not args.no_connect and front.shape[0] >= 2:
                sort_idx = np.argsort(front[:, 0])
                sorted_front = front[sort_idx]
                ax.plot(sorted_front[:, 0], sorted_front[:, 1], color="#ff7f0e", linewidth=1.2, alpha=0.85)

        iter_idx = _extract_iter_index(files[frame_idx])
        ax.set_title(f"{args.title_prefix} | Iteration {iter_idx}")

    anim = FuncAnimation(fig, update, frames=len(files), interval=max(1, int(1000 / max(args.fps, 1))))
    writer = PillowWriter(fps=max(args.fps, 1))
    anim.save(output_path, writer=writer, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved GIF: {output_path}")
    print(f"Frames: {len(files)}")


if __name__ == "__main__":
    main()
