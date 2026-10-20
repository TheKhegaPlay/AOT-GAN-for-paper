"""Multi-rate contextual aggregation blocks used in the HFI-GAN bottleneck."""
import torch
import torch.nn as nn
from torch import Tensor, nn


class ContextualAggregationBlock(nn.Module):
    """Split channels across parallel dilated convolutions, then aggregate."""

    def __init__(self, channels: int, rates: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        super().__init__()
        if channels % len(rates) != 0:
            raise ValueError("channels must be divisible by the number of dilation rates")

        self.branch_channels = channels // len(rates)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    self.branch_channels,
                    self.branch_channels,
                    kernel_size=3,
                    padding=rate,
                    dilation=rate,
                ),
                nn.GELU(),
            )
            for rate in rates
        )
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        chunks = inputs.chunk(len(self.branches), dim=1)
        aggregated = self.fuse(torch.cat(
            [branch(chunk) for branch, chunk in zip(self.branches, chunks)], dim=1
        ))
        return self.activation(inputs + aggregated)