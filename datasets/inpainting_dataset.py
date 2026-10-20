"""Places365 images paired with filtered irregular inpainting masks."""

from __future__ import annotations

import random
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset


_SPLIT_CACHE: dict[tuple[str, str, int, float], tuple[list[str], list[str]]] = {}


class ForensicInpaintingDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Return ``(masked_image, binary_mask, ground_truth)`` tensors.

    The image split is reproducible and independent of the random mask sampled
    for each item. Images are RGB in [-1, 1]; masks are binary with 1=damaged.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        split: Literal["train", "val"] = "train",
        image_size: int = 256,
        validation_fraction: float = 0.05,
        min_coverage: float = 0.05,
        max_coverage: float = 0.20,
        seed: int = 2026,
        mask_fill: float = 1.0,
        image_manifest: str | Path | None = None,
        rebuild_image_manifest: bool = False,
    ) -> None:
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if not 0.0 <= min_coverage < max_coverage <= 1.0:
            raise ValueError("coverage limits must satisfy 0 <= min < max <= 1")

        self.image_size = image_size
        self.mask_fill = mask_fill
        self.split = split
        self.image_dir = Path(image_dir).resolve()
        self.mask_dir = Path(mask_dir)
        manifest_path = Path(image_manifest).resolve() if image_manifest else None
        cache_key = (
            str(self.image_dir),
            str(manifest_path or self.image_dir),
            seed,
            validation_fraction,
        )
        cached_splits = None if rebuild_image_manifest else _SPLIT_CACHE.get(cache_key)
        if cached_splits is None:
            images = self._load_or_scan_images(manifest_path, rebuild_image_manifest)
            random.Random(seed).shuffle(images)
            validation_count = int(len(images) * validation_fraction)
            if validation_fraction > 0 and len(images) > 1:
                validation_count = min(max(1, validation_count), len(images) - 1)
            boundary = len(images) - validation_count
            cached_splits = (images[:boundary], images[boundary:])
            _SPLIT_CACHE[cache_key] = cached_splits
        train_images, validation_images = cached_splits
        selected_images = train_images if split == "train" else validation_images
        if not train_images and not validation_images:
            raise FileNotFoundError(f"No images found recursively under {self.image_dir}")
        self.images = selected_images

        masks: list[Path] = []
        for path in sorted(self.mask_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    binary = image.convert("L").resize(
                        (image_size, image_size), Image.Resampling.NEAREST
                    )
                    coverage = float((np.asarray(binary) >= 128).mean())
                if min_coverage <= coverage <= max_coverage:
                    masks.append(path)
            except (OSError, ValueError):
                continue
        if not masks:
            raise ValueError(
                f"No readable masks in {self.mask_dir} have coverage between "
                f"{min_coverage:.0%} and {max_coverage:.0%} at {image_size}x{image_size}"
            )

        if not self.images:
            raise ValueError(f"The {split} split is empty; adjust validation_fraction or dataset size")
        self.masks = masks

    def _load_or_scan_images(self, manifest_path: Path | None, rebuild: bool) -> list[str]:
        if manifest_path is not None and manifest_path.is_file() and not rebuild:
            print(f"[dataset] Loading image manifest: {manifest_path}", flush=True)
            with manifest_path.open("r", encoding="utf-8") as manifest:
                images = [line.rstrip("\r\n") for line in manifest if line.strip()]
            print(f"[dataset] Loaded {len(images):,} image paths from manifest", flush=True)
            return images

        print(f"[dataset] Scanning image filenames under {self.image_dir}", flush=True)
        started = time.monotonic()
        images: list[str] = []
        for directory, _, filenames in os.walk(self.image_dir):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in self.IMAGE_EXTENSIONS:
                    relative_path = os.path.relpath(
                        os.path.join(directory, filename), self.image_dir
                    ).replace(os.sep, "/")
                    images.append(relative_path)
                    if len(images) % 100_000 == 0:
                        elapsed = time.monotonic() - started
                        print(
                            f"[dataset] Found {len(images):,} images ({elapsed:.1f}s)",
                            flush=True,
                        )

        images.sort()
        if not images:
            return images

        if manifest_path is not None:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = manifest_path.with_name(
                f"{manifest_path.name}.{os.getpid()}.tmp"
            )
            try:
                with temporary_path.open("w", encoding="utf-8", newline="\n") as manifest:
                    manifest.writelines(f"{relative_path}\n" for relative_path in images)
                os.replace(temporary_path, manifest_path)
                print(f"[dataset] Wrote image manifest: {manifest_path}", flush=True)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()

        elapsed = time.monotonic() - started
        print(f"[dataset] Image scan complete: {len(images):,} files in {elapsed:.1f}s", flush=True)
        return images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        image_path = self.image_dir / self.images[index]
        with Image.open(image_path) as source:
            image = ImageOps.fit(
                source.convert("RGB"),
                (self.image_size, self.image_size),
                method=Image.Resampling.BICUBIC,
            )

        mask_path = random.choice(self.masks) if self.split == "train" else self.masks[index % len(self.masks)]
        with Image.open(mask_path) as source_mask:
            mask_image = source_mask.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.NEAREST
            )

        if self.split == "train" and random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask_image = ImageOps.mirror(mask_image)

        image_array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
        mask_array = (np.asarray(mask_image, dtype=np.uint8) >= 128).astype(np.float32)
        ground_truth = torch.from_numpy(image_array).permute(2, 0, 1)
        mask = torch.from_numpy(mask_array).unsqueeze(0)
        masked_image = ground_truth * (1.0 - mask) + self.mask_fill * mask
        return masked_image, mask, ground_truth