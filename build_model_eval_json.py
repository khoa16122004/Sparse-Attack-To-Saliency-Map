import argparse
import json
import os
from pathlib import Path
import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(ROOT_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from util import get_torchvision_model


class ImageNetFolderDataset(Dataset):
    def __init__(self, val_root: Path, synsets: list[str], spatial_transform):
        self.val_root = val_root
        self.synsets = synsets
        self.spatial_transform = spatial_transform
        self.synset_to_index = {synset: idx for idx, synset in enumerate(synsets)}
        self.samples: list[tuple[Path, int, str]] = []

        valid_exts = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}

        for synset in synsets:
            class_dir = val_root / synset
            if not class_dir.exists() or not class_dir.is_dir():
                continue

            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in valid_exts:
                    continue
                self.samples.append((image_path, self.synset_to_index[synset], synset))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label, synset = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.spatial_transform(image)
        relative_posix = image_path.relative_to(self.val_root).as_posix()
        return image_tensor, label, synset, relative_posix


def parse_synsets(synset_words_path: Path) -> list[str]:
    synsets: list[str] = []
    with open(synset_words_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            synset = line.split(" ", 1)[0].strip()
            synsets.append(synset)
    return synsets


def build_emit_path(emit_root: str, relative_posix: str) -> str:
    root = emit_root.rstrip("/")
    return f"{root}/{relative_posix}"


def classify_correct_paths(
    model_name: str,
    imagenet_val_root: Path,
    emit_root: str,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    checkpoint_dir: str | None,
    synset_words_path: Path,
):
    synsets = parse_synsets(synset_words_path)

    model, spatial, normalize = get_torchvision_model(
        model_name=model_name,
        pretrained=True,
        checkpoint_dir=checkpoint_dir,
    )

    if spatial is None or normalize is None:
        raise ValueError("Could not build preprocessing transforms from model weights")

    model = model.to(device)
    model.eval()

    dataset = ImageNetFolderDataset(
        val_root=imagenet_val_root,
        synsets=synsets,
        spatial_transform=spatial,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    correct_paths: dict[str, list[str]] = {synset: [] for synset in synsets}

    with torch.no_grad():
        for images, labels, synset_batch, relative_paths in tqdm(loader, desc=f"Classifying {model_name}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(normalize(images))
            preds = logits.argmax(dim=1)

            matches = preds.eq(labels)
            for i, is_correct in enumerate(matches.tolist()):
                if not is_correct:
                    continue
                synset = synset_batch[i]
                rel_path = relative_paths[i]
                correct_paths[synset].append(build_emit_path(emit_root, rel_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(correct_paths, f, indent=2, ensure_ascii=False)

    total_correct = sum(len(v) for v in correct_paths.values())
    print(f"[OK] model={model_name}")
    print(f"[OK] images_scanned={len(dataset)}")
    print(f"[OK] total_correct={total_correct}")
    print(f"[OK] output={output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify ImageNet val images and save per-class correctly classified paths"
    )
    parser.add_argument("--model", type=str, required=True, help="Torchvision model name (e.g. densenet121)")
    parser.add_argument(
        "--imagenet-val-root",
        type=Path,
        required=True,
        help="Local ImageNet val root that contains class folders",
    )
    parser.add_argument(
        "--emit-root",
        type=str,
        default="/datastore/elo/quanphm/dataset/ImageNet1K/val",
        help="Path prefix used in output JSON values",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file. Default: model_evaluation_results/<model>.json",
    )
    parser.add_argument(
        "--synset-words",
        type=Path,
        default=Path("synset_words.txt"),
        help="Path to synset_words.txt",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing local pretrained checkpoints named <model>.pth",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, fallback to CPU")
        args.device = "cpu"

    output_path = args.output
    if output_path is None:
        output_path = Path("model_evaluation_results") / f"{args.model}.json"

    classify_correct_paths(
        model_name=args.model,
        imagenet_val_root=args.imagenet_val_root,
        emit_root=args.emit_root,
        output_path=output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        synset_words_path=args.synset_words,
    )


if __name__ == "__main__":
    main()
