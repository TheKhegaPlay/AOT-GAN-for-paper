"""Dual-path Hybrid Forensic Inpainting GAN generator."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contextual_agg_block import ContextualAggregationBlock
from .gated_res_block import GatedResidualBlock


class AttentionGate(nn.Module):
    """Use decoder context to select spatial information from an encoder skip."""

    def __init__(self, skip_channels: int, gate_channels: int) -> None:
        super().__init__()
        hidden_channels = max(skip_channels // 2, 8)
        self.skip_projection = nn.Conv2d(skip_channels, hidden_channels, 1)
        self.gate_projection = nn.Conv2d(gate_channels, hidden_channels, 1)
        self.score = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, skip: Tensor, gating: Tensor) -> Tensor:
        gating = F.interpolate(gating, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attention = torch.sigmoid(self.score(torch.relu(
            self.skip_projection(skip) + self.gate_projection(gating)
        )))
        return skip * attention


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel reweighting without normalization."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        return features * self.mlp(self.pool(features))


class HybridForensicGenerator(nn.Module):
    """Restore an image using contextual, detail, and gated skip pathways.

    The main path reasons over a 1/16-resolution context bottleneck. A shallow
    full-resolution branch preserves high-frequency evidence cues; its features
    are pooled to the bottleneck grid and fused through channel attention.
    """

    def __init__(self, base_channels: int = 48, cab_count: int = 6) -> None:
        super().__init__()
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)

        self.encoder = nn.ModuleList([
            GatedResidualBlock(4, channels[0]),
            GatedResidualBlock(channels[0], channels[1]),
            GatedResidualBlock(channels[1], channels[2]),
            GatedResidualBlock(channels[2], channels[3]),
        ])
        self.context = nn.Sequential(*[
            ContextualAggregationBlock(channels[3]) for _ in range(cab_count)
        ])

        self.shallow_path = nn.Sequential(
            nn.Conv2d(4, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, channels[1], 3, padding=1),
            nn.GELU(),
        )
        self.shallow_projection = nn.Conv2d(channels[1], channels[3], 1)
        self.fusion_attention = ChannelAttention(channels[3] * 2)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels[3] * 2, channels[3], 1),
            nn.GELU(),
            nn.Conv2d(channels[3], channels[3], 3, padding=1),
            nn.GELU(),
        )

        self.up_blocks = nn.ModuleList([
            nn.ConvTranspose2d(channels[3], channels[2], 4, stride=2, padding=1),
            nn.ConvTranspose2d(channels[2], channels[1], 4, stride=2, padding=1),
            nn.ConvTranspose2d(channels[1], channels[0], 4, stride=2, padding=1),
            nn.ConvTranspose2d(channels[0], base_channels, 4, stride=2, padding=1),
        ])
        self.skip_gates = nn.ModuleList([
            AttentionGate(channels[2], channels[2]),
            AttentionGate(channels[1], channels[1]),
            AttentionGate(channels[0], channels[0]),
        ])
        self.decoder_fusions = nn.ModuleList([
            self._decoder_fusion(channels[2]),
            self._decoder_fusion(channels[1]),
            self._decoder_fusion(channels[0]),
        ])
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    @staticmethod
    def _decoder_fusion(channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, masked_image: Tensor, mask: Tensor) -> Tensor:
        if masked_image.ndim != 4 or masked_image.shape[1] != 3:
            raise ValueError("masked_image must have shape (B, 3, H, W)")
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must have shape (B, 1, H, W)")
        if masked_image.shape[0] != mask.shape[0] or masked_image.shape[-2:] != mask.shape[-2:]:
            raise ValueError("masked_image and mask batch/spatial dimensions must match")

        inputs = torch.cat((masked_image, mask.to(dtype=masked_image.dtype)), dim=1)
        skips: list[Tensor] = []
        features = inputs
        current_mask = mask
        for block in self.encoder:
            features, current_mask = block(features, current_mask)
            skips.append(features)

        context = self.context(features)
        shallow = self.shallow_path(inputs)
        shallow = F.adaptive_avg_pool2d(shallow, output_size=context.shape[-2:])
        shallow = self.shallow_projection(shallow)
        features = torch.cat((context, shallow), dim=1)
        features = self.fusion_attention(features)
        features = self.fusion(features)

        # Three attention-filtered skips align with encoder levels 3, 2, and 1.
        for index, (up, gate, fusion) in enumerate(zip(
            self.up_blocks[:3], self.skip_gates, self.decoder_fusions
        )):
            features = F.gelu(up(features))
            skip = skips[-2 - index]
            skip = gate(skip, features)
            features = fusion(torch.cat((features, skip), dim=1))

        features = F.gelu(self.up_blocks[3](features))
        return self.output(features)