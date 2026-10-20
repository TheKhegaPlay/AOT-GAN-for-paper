"""Run HFI-GAN on a directory of images and create visual comparisons."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image, ImageOps

from models import HybridForensicGenerator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch image restoration with HFI-GAN")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--selection", choices=("random", "sequential"), default="random")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="save an additional original | masked | restored image for each sample",
    )
    return parser.parse_args()


def image_paths(root: Path) -> Iterator[Path]:
    """Yield supported image paths without calling stat for every directory entry."""
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                yield Path(directory) / filename


def manifest_paths(manifest: Path, image_dir: Path) -> Iterator[Path]:
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            relative_path = line.strip()
            if relative_path:
                yield image_dir / Path(relative_path)


def choose_images(
    candidates: Iterator[Path], count: int, selection: str, rng: random.Random
) -> list[Path]:
    if selection == "sequential":
        selected: list[Path] = []
        for path in candidates:
            selected.append(path)
            if len(selected) == count:
                break
        return selected

    reservoir: list[Path] = []
    seen = 0
    for path in candidates:
        seen += 1
        if len(reservoir) < count:
            reservoir.append(path)
        else:
            index = rng.randrange(seen)
            if index < count:
                reservoir[index] = path
    return reservoir


def load_masks(mask_dir: Path, image_size: int) -> list[tuple[Path, np.ndarray]]:
    masks: list[tuple[Path, np.ndarray]] = []
    for path in image_paths(mask_dir):
        try:
            with Image.open(path) as source:
                mask_image = source.convert("L").resize(
                    (image_size, image_size), Image.Resampling.NEAREST
                )
            mask = (np.asarray(mask_image, dtype=np.uint8) >= 128).astype(np.float32)
            if mask.any():
                masks.append((path, mask))
        except (OSError, ValueError) as exc:
            print(f"[skip mask] {path}: {exc}", flush=True)
    return masks


def image_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def make_mask_overlay(original: Image.Image, mask: np.ndarray) -> Image.Image:
    image = np.asarray(original, dtype=np.float32).copy()
    damaged = mask.astype(bool)
    red = np.zeros_like(image)
    red[..., 0] = 255.0
    image[damaged] = image[damaged] * 0.45 + red[damaged] * 0.55
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")


def make_contact_sheet(images: list[Image.Image]) -> Image.Image:
    width, height = images[0].size
    sheet = Image.new("RGB", (width * len(images), height), color="white")
    for index, image in enumerate(images):
        sheet.paste(image, (index * width, 0))
    return sheet


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("num-samples must be positive")
    if args.image_size <= 0 or args.image_size % 16 != 0:
        raise ValueError("image-size must be positive and divisible by 16")
    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {args.image_dir}")
    if not args.mask_dir.is_dir():
        raise NotADirectoryError(f"Mask directory does not exist: {args.mask_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = HybridForensicGenerator(
        base_channels=int(config.get("base_channels", 48)),
        cab_count=int(config.get("cab_count", 6)),
    ).to(device)
    model.load_state_dict(checkpoint["generator"])
    model.eval()

    image_dir = args.image_dir.resolve()
    manifest = args.checkpoint.parent / "places365_images.txt"
    configured_image_dir = config.get("image_dir")
    use_manifest = (
        manifest.is_file()
        and configured_image_dir is not None
        and Path(configured_image_dir).resolve() == image_dir
    )
    if use_manifest:
        print(f"[images] Using cached manifest: {manifest}", flush=True)
        candidates = manifest_paths(manifest, image_dir)
    else:
        print(f"[images] Scanning {image_dir}", flush=True)
        candidates = image_paths(image_dir)
    selected = choose_images(candidates, args.num_samples, args.selection, rng)
    if not selected:
        raise FileNotFoundError(f"No JPG/PNG images found under {image_dir}")

    masks = load_masks(args.mask_dir, args.image_size)
    if not masks:
        raise ValueError(f"No readable, non-empty JPG/PNG masks found under {args.mask_dir}")

    print(f"[setup] {len(selected)} candidate images, {len(masks)} readable masks, device={device}", flush=True)
    processed = 0
    skipped = 0
    with torch.inference_mode():
        for sample_index, image_path in enumerate(selected, start=1):
            try:
                with Image.open(image_path) as source:
                    original = ImageOps.fit(
                        source.convert("RGB"),
                        (args.image_size, args.image_size),
                        method=Image.Resampling.BICUBIC,
                    )
            except (OSError, ValueError) as exc:
                skipped += 1
                print(f"[{sample_index}/{len(selected)}] skip unreadable image {image_path}: {exc}", flush=True)
                continue

            mask_path, mask_array = rng.choice(masks)
            original_tensor = image_tensor(original, device)
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(device)
            masked_input = original_tensor * (1.0 - mask_tensor) + mask_tensor
            prediction = model(masked_input, mask_tensor)
            restored = prediction * mask_tensor + original_tensor * (1.0 - mask_tensor)
            restored_array = restored[0].clamp(-1, 1).add(1).mul(127.5)
            restored_array = restored_array.byte().permute(1, 2, 0).cpu().numpy()
            restored_image = Image.fromarray(restored_array, mode="RGB")
            masked_visual = make_mask_overlay(original, mask_array)

            output_index = processed + 1
            original.save(args.output_dir / f"original_{output_index:04d}.jpg", quality=95)
            masked_visual.save(args.output_dir / f"masked_{output_index:04d}.jpg", quality=95)
            restored_image.save(args.output_dir / f"restored_{output_index:04d}.jpg", quality=95)
            if args.contact_sheet:
                make_contact_sheet([original, masked_visual, restored_image]).save(
                    args.output_dir / f"comparison_{output_index:04d}.jpg", quality=95
                )

            processed += 1
            print(
                f"[{sample_index}/{len(selected)}] saved sample {output_index:04d}: "
                f"{image_path.name} + {mask_path.name}",
                flush=True,
            )

    print(
        f"Done: processed {processed}, skipped {skipped}. Results: {args.output_dir.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()