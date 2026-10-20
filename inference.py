"""Restore one image using an HFI-GAN checkpoint and a binary damage mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from models import HybridForensicGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HFI-GAN on one image and mask")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    image_size = args.image_size or int(config.get("image_size", 256))
    model = HybridForensicGenerator(
        base_channels=int(config.get("base_channels", 48)),
        cab_count=int(config.get("cab_count", 6)),
    ).to(device)
    model.load_state_dict(checkpoint["generator"])
    model.eval()

    with Image.open(args.image) as source:
        image = ImageOps.fit(
            source.convert("RGB"), (image_size, image_size), method=Image.Resampling.BICUBIC
        )
    with Image.open(args.mask) as source_mask:
        mask_image = source_mask.convert("L").resize(
            (image_size, image_size), Image.Resampling.NEAREST
        )

    image_array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    mask_array = (np.asarray(mask_image, dtype=np.uint8) >= 128).astype(np.float32)
    ground_truth = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(device)
    masked_image = ground_truth * (1.0 - mask) + mask

    with torch.inference_mode():
        prediction = model(masked_image, mask)
        restored = prediction * mask + ground_truth * (1.0 - mask)

    output_array = restored[0].clamp(-1, 1).add(1).mul(127.5).byte()
    output_array = output_array.permute(1, 2, 0).cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_array, mode="RGB").save(args.output)
    print(f"Restored image saved to {args.output}")


if __name__ == "__main__":
    main()