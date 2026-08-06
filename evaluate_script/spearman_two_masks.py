import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


def rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    i = 0
    n = sorted_vals.size
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1

        avg_rank = (i + 1 + j) * 0.5
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.size != y.size:
        raise ValueError("Spearman inputs must have same number of elements")
    if x.size < 2:
        return float("nan")

    rx = rankdata_average_ties(x)
    ry = rankdata_average_ties(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = math.sqrt(float(np.dot(rx, rx)) * float(np.dot(ry, ry)))
    if denom < 1e-12:
        return float("nan")

    return float(np.dot(rx, ry) / denom)


def load_gray_image_array(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    return np.asarray(image, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Spearman correlation between two mask images")
    parser.add_argument("--img1", type=str, required=True, help="Path to first image (PNG mask)")
    parser.add_argument("--img2", type=str, required=True, help="Path to second image (PNG mask)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    img1_path = Path(args.img1)
    img2_path = Path(args.img2)

    if not img1_path.exists():
        raise FileNotFoundError(f"Image not found: {img1_path}")
    if not img2_path.exists():
        raise FileNotFoundError(f"Image not found: {img2_path}")

    img1 = load_gray_image_array(img1_path)
    img2 = load_gray_image_array(img2_path)

    if img1.shape != img2.shape:
        raise ValueError(
            f"Image shapes differ: img1={img1.shape}, img2={img2.shape}. "
            "Please resize/crop to the same size before computing Spearman."
        )

    corr = spearman_rank_corr(img1, img2)
    print(f"spearman_corr: {corr:.10f}")


if __name__ == "__main__":
    main()
