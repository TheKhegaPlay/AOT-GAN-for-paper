"""Evaluate an HFI-GAN checkpoint with PSNR/SSIM on restored images."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
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
    parser.add_argument("--image-manifest", type=Path, default=None,
                        help="use a saved, sorted image list for a reproducible split")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="evaluate only the first N validation batches")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="write evaluation metadata and metrics to a JSON file")
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
        min_coverage=float(config.get("mask_min_coverage", 0.05)),
        max_coverage=float(config.get("mask_max_coverage", 0.20)),
        image_manifest=args.image_manifest,
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
    evaluated_images = 0
    with torch.inference_mode():
        for batch_index, (masked, mask, target) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            masked, mask, target = masked.to(device), mask.to(device), target.to(device)
            prediction = model(masked, mask)
            completed = prediction * mask + target * (1.0 - mask)
            full_psnr.append(psnr(completed, target).item())
            full_ssim.append(ssim(completed, target).item())
            damaged_l1.append(masked_l1(prediction, target, mask).item())
            evaluated_images += target.shape[0]
            if lpips_metric is not None:
                lpips_values.append(lpips_metric(completed, target).mean().item())

    if not evaluated_images:
        raise ValueError("No validation images were evaluated; check --max-batches and dataset")

    metrics = {
        "full_image_psnr_db": sum(full_psnr) / len(full_psnr),
        "full_image_ssim": sum(full_ssim) / len(full_ssim),
        "damaged_region_l1": sum(damaged_l1) / len(damaged_l1),
    }
    if lpips_values:
        metrics["lpips"] = sum(lpips_values) / len(lpips_values)

    checkpoint_path = args.checkpoint.resolve()
    try:
        checkpoint_reference = checkpoint_path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        checkpoint_reference = checkpoint_path.name
    with checkpoint_path.open("rb") as checkpoint_file:
        checkpoint_sha256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()

    result = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_reference,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_iteration": int(checkpoint.get("iteration", 0)),
        "model": {
            "name": "HFI-GAN",
            "base_channels": int(config.get("base_channels", 48)),
            "cab_count": int(config.get("cab_count", 6)),
        },
        "dataset": {
            "name": "Places365 Standard",
            "image_size": image_size,
            "validation_fraction": args.validation_fraction,
            "validation_seed": args.seed,
            "validation_split_images": len(dataset),
            "image_manifest": (
                args.image_manifest.name if args.image_manifest is not None else None
            ),
            "mask_directory": args.mask_dir.name,
            "eligible_masks": len(dataset.masks),
            "mask_coverage_range": [
                float(config.get("mask_min_coverage", 0.05)),
                float(config.get("mask_max_coverage", 0.20)),
            ],
        },
        "protocol": {
            "evaluated_images": evaluated_images,
            "batch_size": args.batch_size,
            "batches": len(full_psnr),
            "max_batches": args.max_batches,
            "subset": args.max_batches is not None,
            "metrics_aggregation": "arithmetic mean of batch means",
            "compositing": "ground truth outside mask; model prediction inside mask",
        },
        "metrics": metrics,
    }

    print(f"Images evaluated: {evaluated_images} / {len(dataset)}")
    print(f"PSNR (full image): {metrics['full_image_psnr_db']:.4f} dB")
    print(f"SSIM (full image): {metrics['full_image_ssim']:.6f}")
    print(f"L1 (damaged regions): {metrics['damaged_region_l1']:.6f}")
    if "lpips" in metrics:
        print(f"LPIPS: {metrics['lpips']:.6f}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"JSON results: {args.output_json}")


if __name__ == "__main__":
    main()
