"""Train the original HFI-GAN architecture from scratch or from a checkpoint."""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from datasets import ForensicInpaintingDataset
from models import ConditionalPatchDiscriminator, HybridForensicGenerator
from models.losses import (
    VGG19FeatureLoss,
    gram_matrix,
    masked_l1,
    masked_patch_weights,
    psnr,
    ssim,
    weighted_patch_mean,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hybrid Forensic Inpainting GAN")
    parser.add_argument("--image-dir", type=Path, default=Path("data/places365_standard"))
    parser.add_argument("--mask-dir", type=Path, default=Path("data/masks/nvidia"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/hfi_gan"))
    parser.add_argument("--image-manifest", type=Path, default=None,
                        help="cache of image paths; defaults to checkpoint-dir/places365_images.txt")
    parser.add_argument("--rebuild-image-manifest", action="store_true",
                        help="rescan the image directory and replace the cached manifest")
    parser.add_argument("--iterations", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate-g", type=float, default=1e-4)
    parser.add_argument("--learning-rate-d", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--cab-count", type=int, default=6)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--perceptual-weight", type=float, default=0.1)
    parser.add_argument("--adversarial-weight", type=float, default=0.01)
    parser.add_argument(
        "--style-weight", type=float, default=25.0,
        help="Gram style loss weight; set to 0 to disable the style branch",
    )
    parser.add_argument("--gan-loss", choices=("hinge", "lsgan"), default="hinge")
    parser.add_argument("--no-pretrained-vgg", action="store_true",
                        help="initialize VGG-19 randomly instead of loading ImageNet weights")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--mask-min-coverage", type=float, default=0.05)
    parser.add_argument("--mask-max-coverage", type=float, default=0.20)
    parser.add_argument("--validation-every", type=int, default=1000)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def initialize_distributed() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return distributed, rank, local_rank, world_size, device


def seed_everything(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generator_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def validate(
    generator: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[float, float]:
    generator = generator_model(generator)
    generator.eval()
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    with torch.no_grad():
        for batch_index, (masked, mask, target) in enumerate(loader):
            if batch_index >= max_batches:
                break
            masked, mask, target = masked.to(device), mask.to(device), target.to(device)
            prediction = generator(masked, mask)
            completed = prediction * mask + target * (1.0 - mask)
            psnr_values.append(psnr(completed, target).item())
            ssim_values.append(ssim(completed, target).item())
    generator.train()
    if not psnr_values:
        return float("nan"), float("nan")
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


def checkpoint_state(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    iteration: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    discriminator_module = discriminator.module if isinstance(
        discriminator, DistributedDataParallel
    ) else discriminator
    return {
        "iteration": iteration,
        "generator": generator_model(generator).state_dict(),
        "discriminator": discriminator_module.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "config": vars(args),
    }


def save_checkpoint(state: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(state, temporary)
        for attempt in range(3):
            try:
                os.replace(temporary, path)
                return path
            except PermissionError:
                if attempt == 2:
                    break
                time.sleep(0.5 * (attempt + 1))

        fallback = path.with_name(
            f"{path.stem}_fallback_{os.getpid()}_{time.time_ns()}{path.suffix}"
        )
        os.replace(temporary, fallback)
        print(
            f"[checkpoint] Target is locked; saved checkpoint to {fallback}",
            flush=True,
        )
        return fallback
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    if args.image_size % 16 != 0:
        raise ValueError("image-size must be divisible by 16 for four downsampling stages")
    if args.iterations <= 0 or args.batch_size <= 0:
        raise ValueError("iterations and batch-size must be positive")
    if min(args.log_every, args.checkpoint_every) <= 0:
        raise ValueError("log-every and checkpoint-every must be positive")
    if args.validation_every <= 0 or args.validation_batches <= 0:
        raise ValueError("validation-every and validation-batches must be positive")

    distributed, rank, local_rank, _, device = initialize_distributed()
    seed_everything(args.seed, rank)

    image_manifest = args.image_manifest or args.checkpoint_dir / "places365_images.txt"
    if distributed and rank != 0:
        dist.barrier()
    train_dataset = ForensicInpaintingDataset(
        args.image_dir, args.mask_dir, "train", args.image_size,
        args.validation_fraction,
        min_coverage=args.mask_min_coverage,
        max_coverage=args.mask_max_coverage,
        seed=args.seed,
        image_manifest=image_manifest,
        rebuild_image_manifest=args.rebuild_image_manifest,
    )
    validation_dataset = ForensicInpaintingDataset(
        args.image_dir, args.mask_dir, "val", args.image_size,
        args.validation_fraction,
        min_coverage=args.mask_min_coverage,
        max_coverage=args.mask_max_coverage,
        seed=args.seed,
        image_manifest=image_manifest,
    ) if args.validation_fraction > 0 and len(train_dataset.images) > 1 else None
    if distributed and rank == 0:
        dist.barrier()
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_sampler = DistributedSampler(validation_dataset, shuffle=False) if distributed else None
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=validation_sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty; reduce batch-size or add more training images")

    generator: torch.nn.Module = HybridForensicGenerator(
        base_channels=args.base_channels, cab_count=args.cab_count
    ).to(device)
    discriminator: torch.nn.Module = ConditionalPatchDiscriminator().to(device)
    if distributed:
        generator = DistributedDataParallel(
            generator,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
        discriminator = DistributedDataParallel(
            discriminator,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )

    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=args.learning_rate_g, betas=(args.beta1, args.beta2)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=args.learning_rate_d, betas=(args.beta1, args.beta2)
    )
    perceptual = VGG19FeatureLoss(pretrained=not args.no_pretrained_vgg).to(device)
    perceptual.eval()
    if rank == 0:
        style_status = "enabled" if args.style_weight > 0 else "disabled"
        print(f"[loss] Gram style loss {style_status}, weight={args.style_weight:g}", flush=True)
    start_iteration = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        generator_model(generator).load_state_dict(checkpoint["generator"])
        discriminator_module = discriminator.module if isinstance(
            discriminator, DistributedDataParallel
        ) else discriminator
        discriminator_module.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_iteration = int(checkpoint["iteration"])

    iterator = iter(train_loader)
    sampler_epoch = 0
    for iteration in range(start_iteration + 1, args.iterations + 1):
        try:
            masked, mask, target = next(iterator)
        except StopIteration:
            sampler_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(sampler_epoch)
            iterator = iter(train_loader)
            masked, mask, target = next(iterator)
        masked, mask, target = masked.to(device), mask.to(device), target.to(device)

        fake = generator(masked, mask)
        completed = fake * mask + target * (1.0 - mask)
        optimizer_d.zero_grad(set_to_none=True)
        real_logits = discriminator(target, mask)
        fake_logits = discriminator(completed.detach(), mask)
        patch_weights = masked_patch_weights(mask, real_logits)
        if args.gan_loss == "hinge":
            d_loss = weighted_patch_mean(F.relu(1.0 - real_logits), patch_weights)
            d_loss += weighted_patch_mean(F.relu(1.0 + fake_logits), patch_weights)
        else:
            d_loss = weighted_patch_mean((real_logits - 1.0).square(), patch_weights)
            d_loss += weighted_patch_mean(fake_logits.square(), patch_weights)
        d_loss.backward()
        optimizer_d.step()

        discriminator_for_generator = discriminator.module if isinstance(
            discriminator, DistributedDataParallel
        ) else discriminator
        for parameter in discriminator_for_generator.parameters():
            parameter.requires_grad_(False)
        optimizer_g.zero_grad(set_to_none=True)
        completed = fake * mask + target * (1.0 - mask)
        l1_loss = masked_l1(fake, target, mask)
        perceptual_loss = perceptual(completed, target)
        generator_logits = discriminator_for_generator(completed, mask)
        generator_weights = masked_patch_weights(mask, generator_logits)
        if args.gan_loss == "hinge":
            adversarial_loss = -weighted_patch_mean(generator_logits, generator_weights)
        else:
            adversarial_loss = weighted_patch_mean(
                (generator_logits - 1.0).square(), generator_weights
            )

        style_loss = fake.new_zeros(())
        if args.style_weight > 0:
            fake_features = perceptual._extract(completed)
            with torch.no_grad():
                target_features = perceptual._extract(target)
            style_loss = sum(
                F.l1_loss(gram_matrix(predicted), gram_matrix(truth))
                for predicted, truth in zip(fake_features, target_features)
            ) / len(fake_features)
        g_loss = (
            args.l1_weight * l1_loss
            + args.perceptual_weight * perceptual_loss
            + args.adversarial_weight * adversarial_loss
            + args.style_weight * style_loss
        )
        g_loss.backward()
        optimizer_g.step()
        for parameter in discriminator_for_generator.parameters():
            parameter.requires_grad_(True)

        if rank == 0 and iteration % args.log_every == 0:
            print(
                f"iter={iteration}/{args.iterations} G={g_loss.item():.4f} "
                f"D={d_loss.item():.4f} L1={l1_loss.item():.4f} "
                f"VGG={perceptual_loss.item():.4f} adv={adversarial_loss.item():.4f} "
                f"style={style_loss.item():.6f} "
                f"style_weighted={(args.style_weight * style_loss).item():.4f}",
                flush=True,
            )

        if validation_loader is not None and iteration % args.validation_every == 0:
            if not distributed or rank == 0:
                val_psnr, val_ssim = validate(
                    generator, validation_loader, device, args.validation_batches
                )
                print(f"validation iter={iteration} PSNR={val_psnr:.3f} SSIM={val_ssim:.4f}")
            if distributed:
                dist.barrier()
                generator.train()

        if rank == 0 and iteration % args.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_state(generator, discriminator, optimizer_g, optimizer_d, iteration, args),
                args.checkpoint_dir / f"hfi_gan_{iteration:08d}.pt",
            )

    if rank == 0:
        save_checkpoint(
            checkpoint_state(generator, discriminator, optimizer_g, optimizer_d, args.iterations, args),
            args.checkpoint_dir / "hfi_gan_final.pt",
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
