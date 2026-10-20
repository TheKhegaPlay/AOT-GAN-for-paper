"""Losses and image-quality metrics used by HFI-GAN."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def masked_l1(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean absolute reconstruction error over damaged pixels only."""
    expanded_mask = mask.expand_as(prediction)
    return ((prediction - target).abs() * expanded_mask).sum() / expanded_mask.sum().clamp_min(1.0)


class VGG19FeatureLoss(nn.Module):
    """Frozen VGG-19 feature distance at relu1_2, relu2_2, relu3_3, relu4_3."""

    FEATURE_INDICES = (3, 8, 15, 24)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import VGG19_Weights, vgg19
        except Exception as exc:
            raise RuntimeError("VGG perceptual loss requires a working torchvision installation") from exc

        weights = VGG19_Weights.DEFAULT if pretrained else None
        self.features = vgg19(weights=weights).features[:25]
        self.features.eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> "VGG19FeatureLoss":
        super().train(False)
        self.features.eval()
        return self

    def _extract(self, image: Tensor) -> list[Tensor]:
        image = ((image.clamp(-1, 1) + 1.0) / 2.0 - self.mean) / self.std
        outputs = []
        for index, layer in enumerate(self.features):
            image = layer(image)
            if index in self.FEATURE_INDICES:
                outputs.append(image)
        return outputs

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        predicted_features = self._extract(prediction)
        with torch.no_grad():
            target_features = self._extract(target)
        return sum(F.l1_loss(predicted, truth) for predicted, truth in zip(
            predicted_features, target_features
        )) / len(self.FEATURE_INDICES)


def gram_matrix(features: Tensor) -> Tensor:
    batch, channels, height, width = features.shape
    flattened = features.reshape(batch, channels, height * width)
    return flattened @ flattened.transpose(1, 2) / (channels * height * width)


def masked_patch_weights(mask: Tensor, prediction: Tensor, outside_weight: float = 0.1) -> Tensor:
    patch_mask = F.interpolate(mask, size=prediction.shape[-2:], mode="area")
    return outside_weight + (1.0 - outside_weight) * patch_mask


def weighted_patch_mean(values: Tensor, weights: Tensor) -> Tensor:
    weights = weights.expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def psnr(prediction: Tensor, target: Tensor) -> Tensor:
    """Batch-mean PSNR for images represented in [-1, 1]."""
    mse = (prediction - target).square().flatten(1).mean(dim=1).clamp_min(1e-10)
    return (10.0 * torch.log10(4.0 / mse)).mean()


def ssim(prediction: Tensor, target: Tensor, window_size: int = 11) -> Tensor:
    """Differentiable local SSIM metric, returned as a batch mean."""
    prediction = (prediction.clamp(-1, 1) + 1.0) / 2.0
    target = (target.clamp(-1, 1) + 1.0) / 2.0
    padding = window_size // 2
    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(prediction.square(), window_size, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), window_size, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, window_size, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-12)
    return score.flatten(1).mean(dim=1).mean()