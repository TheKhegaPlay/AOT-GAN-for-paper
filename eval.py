"""Evaluate an HFI-GAN checkpoint with PSNR/SSIM on restored images."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import ForensicInpaintingDataset
from models import HybridForensicGenerator
from models.losses import masked_l1, psnr, ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained HFI-GAN checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lpips", action="store_true", help="also compute LPIPS if installed")
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

    dataset = ForensicInpaintingDataset(
        args.image_dir, args.mask_dir, split="val", image_size=image_size,
        validation_fraction=args.validation_fraction, seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    lpips_metric = None
    if args.lpips:
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError("Install the optional dependency with: pip install lpips") from exc
        lpips_metric = lpips.LPIPS(net="alex").to(device).eval()

    full_psnr: list[float] = []
    full_ssim: list[float] = []
    damaged_l1: list[float] = []
    lpips_values: list[float] = []
    with torch.inference_mode():
        for masked, mask, target in loader:
            masked, mask, target = masked.to(device), mask.to(device), target.to(device)
            prediction = model(masked, mask)
            completed = prediction * mask + target * (1.0 - mask)
            full_psnr.append(psnr(completed, target).item())
            full_ssim.append(ssim(completed, target).item())
            damaged_l1.append(masked_l1(prediction, target, mask).item())
            if lpips_metric is not None:
                lpips_values.append(lpips_metric(completed, target).mean().item())

    print(f"Images evaluated: {len(dataset)}")
    print(f"PSNR (full image): {sum(full_psnr) / len(full_psnr):.4f} dB")
    print(f"SSIM (full image): {sum(full_ssim) / len(full_ssim):.6f}")
    print(f"L1 (damaged regions): {sum(damaged_l1) / len(damaged_l1):.6f}")
    if lpips_values:
        print(f"LPIPS: {sum(lpips_values) / len(lpips_values):.6f}")


if __name__ == "__main__":
    main()
