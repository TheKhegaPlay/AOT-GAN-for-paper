"""Conditional PatchGAN discriminator for masked image regions."""

import torch
from torch import Tensor, nn
from torch.nn.utils import spectral_norm


class ConditionalPatchDiscriminator(nn.Module):
    """Classify local patches from image+mask pairs, returning a spatial map."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            spectral_norm(nn.Conv2d(4, base_channels, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, 3, stride=1, padding=1),
        )

    def forward(self, image: Tensor, mask: Tensor) -> Tensor:
        if image.shape[0] != mask.shape[0] or image.shape[-2:] != mask.shape[-2:]:
            raise ValueError("image and mask batch/spatial dimensions must match")
        return self.network(torch.cat((image, mask.to(image.dtype)), dim=1))