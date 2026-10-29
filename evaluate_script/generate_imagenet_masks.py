"""Generate text-prompted SAM3 pseudo-GT masks in the ImageNet directory layout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompts(path, synset_path):
    prompts = {}
    for line in synset_path.read_text(encoding="utf-8").splitlines():
        synset, names = line.split(" ", 1)
        prompts[synset] = names.split(",")[0].strip()
    if path:
        overrides = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) or not value.strip()
            for key, value in overrides.items()
        ):
            raise ValueError(f"{path}: expected a JSON object mapping class IDs to nonempty prompts")
        prompts.update({key: value.strip() for key, value in overrides.items()})
    return prompts


def union_masks(masks, height, width):
    array = masks.detach().cpu().numpy()
    if array.size == 0:
        return np.zeros((height, width), dtype=bool)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3 or array.shape[1:] != (height, width):
        raise ValueError(f"SAM masks have shape {array.shape}, expected (N, {height}, {width})")
    if not np.isfinite(array).all():
        raise ValueError("SAM masks contain non-finite values")
    return np.any(array > 0.5, axis=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet-root", type=Path, required=True,
                        help="Directory directly containing class folders")
    parser.add_argument("--output-dir", type=Path, default=Path("imagenet_mask"))
    parser.add_argument("--prompts-json", type=Path)
    parser.add_argument("--synset-words", type=Path, default=ROOT / "synset_words.txt")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=Path, help="Local SAM3 checkpoint; otherwise use SAM3's hub default")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be in [0, 1]")
    source = args.imagenet_root.resolve()
    output = args.output_dir.resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if source == output or source in output.parents or output in source.parents:
        parser.error("Image and mask roots must be separate, non-nested directories")
    prompts = load_prompts(args.prompts_json, args.synset_words)
    images = sorted(path for path in source.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No images found in {source}")
    destinations = set()
    for path in images:
        relative = path.relative_to(source)
        if len(relative.parts) < 2 or relative.parts[0] not in prompts:
            raise ValueError(f"No class prompt for {relative}; provide --prompts-json")
        destination = output / relative.with_suffix(".png")
        if destination in destinations:
            raise ValueError(f"Multiple images map to {destination}")
        destinations.add(destination)
        if destination.exists() and not args.replace:
            raise FileExistsError(f"{destination}: use --replace to regenerate")
    manifest_path = output / "mask_manifest.json"
    if manifest_path.exists() and not args.replace:
        raise FileExistsError(f"{manifest_path}: use --replace to regenerate")

    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; explicitly select --device cpu")
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("SAM3 CUDA inference requires a GPU with BF16 support")
    kwargs = {"device": args.device}
    if args.checkpoint:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        kwargs.update(checkpoint_path=str(args.checkpoint), load_from_HF=False)
    model = build_sam3_image_model(**kwargs).eval()
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold,
                             device=args.device)
    manifest = {
        "schema_version": 1, "status": "in_progress", "source_root": str(source),
        "protocol": {
            "model": "sam3", "checkpoint": str(args.checkpoint) if args.checkpoint else "sam3_hub_default",
            "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
            "confidence_threshold": args.confidence_threshold,
            "instance_reduction": "union", "mask_threshold": 0.5,
            "coordinates": "original_image", "foreground_value": 255,
            "ground_truth": "text_prompted_pseudo_gt",
            "precision": "cuda_bfloat16_autocast" if args.device == "cuda" else "cpu_default",
        },
        "images": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    # Invalidate any previous completed manifest before overwriting its masks.
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    empty_count = 0
    with torch.inference_mode():
        for path in images:
            relative = path.relative_to(source)
            prompt = prompts[relative.parts[0]]
            with Image.open(path) as image:
                image = image.convert("RGB")
                width, height = image.size
                with torch.autocast(
                    device_type=args.device, dtype=torch.bfloat16,
                    enabled=args.device == "cuda",
                ):
                    state = processor.set_image(image)
                    result = processor.set_text_prompt(state=state, prompt=prompt)
            mask = union_masks(result["masks"], height, width)
            destination = output / relative.with_suffix(".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(destination)
            empty = not mask.any()
            empty_count += int(empty)
            manifest["images"][relative.as_posix()] = {
                "mask": relative.with_suffix(".png").as_posix(), "prompt": prompt,
                "source_sha256": sha256(path), "mask_sha256": sha256(destination),
                "shape": [height, width], "foreground_pixels": int(mask.sum()),
                "empty": empty,
            }
            print(f"[{'EMPTY' if empty else 'OK'}] {relative} | prompt={prompt!r}", flush=True)
    manifest.update(status="complete", empty_masks=empty_count)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[DONE] {len(images)} masks; {empty_count} empty (review before using as pseudo-GT)")


if __name__ == "__main__":
    main()
