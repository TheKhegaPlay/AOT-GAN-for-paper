import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .common import BaseNetwork


class ConvBlock(nn.Module):
    """
    Стандартный блок convolution с опциональной нормализацией и dropout.
    Используется в decoder для обработки skip-соединений.
    """
    def __init__(self, in_channels, out_channels, use_norm=True, dropout=0.0):
        super(ConvBlock, self).__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True),
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=False))
        layers.append(nn.ReLU(True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpConv(nn.Module):
    """Upsampling + convolution блок."""
    def __init__(self, in_channels, out_channels, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=True)
        return self.conv(x)


class AOTBlock(nn.Module):
    """
    Aggregated Contextual Transformation Block.
    Агрегирует информацию из разных receptive fields через гейтинговый механизм.
    """
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                "block{}".format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate), nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate), nn.ReLU(True)
                ),
            )
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f"block{str(i).zfill(2)}")(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    """Layer normalization с масштабированием (используется в AOT gate)."""
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


class HybridInpaintGenerator(BaseNetwork):
    """
    Hybrid Forensic Inpainting Generator (HFI-Gen).
    
    Архитектура:
    - Truncated U-Net style encoder-decoder (4 уровня)
    - Сильные skip-connections для сохранения мелких деталей (текст, грани, следы)
    - AOT blocks в bottleneck для агрегации контекста с multiple receptive fields
    - Параллельный Confidence Estimation Branch с отдельным decoder
    
    Выходы:
    - restored_image: восстановленное изображение (B, 3, H, W), tanh normalized
    - confidence_map: карта уверенности (B, 1, H, W), sigmoid normalized [0, 1]
    
    Это позволяет forensic экспертам понять, в каких регионах восстановление надёжно.
    """

    def __init__(self, args):
        super(HybridInpaintGenerator, self).__init__()

        # ============ ENCODER ============
        # Первый слой: 4 канала (RGB + mask) -> 64
        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, kernel_size=7, stride=1, padding=0, bias=True),
            nn.ReLU(True),
        )

        # Второй уровень: 64 -> 128, stride=2
        self.enc2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(128, affine=False),
            nn.ReLU(True),
        )

        # Третий уровень: 128 -> 256, stride=2
        self.enc3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(256, affine=False),
            nn.ReLU(True),
        )

        # Четвёртый уровень: 256 -> 512, stride=2 (bottleneck)
        self.enc4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(512, affine=False),
            nn.ReLU(True),
        )

        # ============ BOTTLENECK ============
        # AOT blocks: множественные receptive fields для контекстной агрегации
        self.middle = nn.Sequential(*[AOTBlock(512, args.rates) for _ in range(args.block_num)])

        # ============ DECODER (для восстановленного изображения) ============
        # Уровень 4: 512 -> 256
        self.up4 = UpConv(512, 256)
        self.dec4 = ConvBlock(256 + 256, 256, dropout=0.3)  # +256 от skip e3

        # Уровень 3: 256 -> 128
        self.up3 = UpConv(256, 128)
        self.dec3 = ConvBlock(128 + 128, 128, dropout=0.3)  # +128 от skip e2

        # Уровень 2: 128 -> 64
        self.up2 = UpConv(128, 64)
        self.dec2 = ConvBlock(64 + 64, 64, dropout=0.3)  # +64 от skip e1

        # Финальный слой: 64 -> 3
        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
        )

        # ============ CONFIDENCE BRANCH (отдельный decoder) ============
        # Параллельная ветка для оценки уверенности в восстановлении
        # использует те же skip-соединения, но с отдельными весами
        
        self.conf_up4 = UpConv(512, 256)
        self.conf_dec4 = ConvBlock(256 + 256, 256, dropout=0.0)

        self.conf_up3 = UpConv(256, 128)
        self.conf_dec3 = ConvBlock(128 + 128, 128, dropout=0.0)

        self.conf_up2 = UpConv(128, 64)
        self.conf_dec2 = ConvBlock(64 + 64, 64, dropout=0.0)

        # Выход: 1 канал (confidence score), sigmoid для [0, 1]
        self.conf_out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid(),
        )

        self.init_weights()

    def forward(self, x, mask):
        """
        Args:
            x: (B, 3, H, W) - input image
            mask: (B, 1, H, W) - inpainting mask (1 = damaged region, 0 = valid)
        
        Returns:
            restored: (B, 3, H, W) - restored image in [-1, 1]
            confidence_map: (B, 1, H, W) - confidence scores in [0, 1]
        """
        # Конкатенируем RGB + mask
        x = torch.cat([x, mask], dim=1)  # (B, 4, H, W)

        # ========== ENCODER PATH ==========
        e1 = self.enc1(x)    # (B, 64, H, W)
        e2 = self.enc2(e1)   # (B, 128, H/2, W/2)
        e3 = self.enc3(e2)   # (B, 256, H/4, W/4)
        e4 = self.enc4(e3)   # (B, 512, H/8, W/8)

        # ========== BOTTLENECK ==========
        m = self.middle(e4)  # (B, 512, H/8, W/8) - агрегированный контекст

        # ========== MAIN DECODER (для восстановленного изображения) ==========
        d4 = self.up4(m)                        # (B, 256, H/4, W/4)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))  # skip-connection от e3

        d3 = self.up3(d4)                       # (B, 128, H/2, W/2)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))  # skip-connection от e2

        d2 = self.up2(d3)                       # (B, 64, H, W)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))  # skip-connection от e1

        restored = self.final(d2)  # (B, 3, H, W) в [-1, 1]

        # ========== CONFIDENCE BRANCH (параллельный decoder) ==========
        c4 = self.conf_up4(m)
        c4 = self.conf_dec4(torch.cat([c4, e3], dim=1))

        c3 = self.conf_up3(c4)
        c3 = self.conf_dec3(torch.cat([c3, e2], dim=1))

        c2 = self.conf_up2(c3)
        c2 = self.conf_dec2(torch.cat([c2, e1], dim=1))

        confidence_map = self.conf_out(c2)  # (B, 1, H, W) в [0, 1]

        return restored, confidence_map


# ==== BACKWARD COMPATIBILITY ====
# Сохраняем оригинальный InpaintGenerator как алиас для новой архитектуры
InpaintGenerator = HybridInpaintGenerator


# ----- discriminator -----
class Discriminator(BaseNetwork):
    def __init__(
        self,
    ):
        super(Discriminator, self).__init__()
        inc = 3
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(inc, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

        self.init_weights()

    def forward(self, x):
        feat = self.conv(x)
        return feat
