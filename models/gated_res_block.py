"""Mask-aware gated residual downsampling blocks for HFI-GAN."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GatedResidualBlock(nn.Module):
    """Downsample features and gate the residual update spatially.

    The mask is resized alongside the features. Updates are attenuated on known
    pixels and remain fully available inside damaged regions.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.gate = nn.Conv2d(out_channels + 1, out_channels, 3, padding=1)
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        shortcut = self.shortcut(inputs)
        residual = self.residual(inputs)
        mask = F.interpolate(mask, size=residual.shape[-2:], mode="nearest")
        gate = torch.sigmoid(self.gate(torch.cat((residual, mask), dim=1)))
        update_scale = 0.1 + 0.9 * mask
        return self.activation(shortcut + residual * gate * update_scale), mask