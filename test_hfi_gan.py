"""Quick 256x256 forward/backward smoke test for HFI-GAN."""

import torch

from models import ConditionalPatchDiscriminator, HybridForensicGenerator


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = HybridForensicGenerator(base_channels=8, cab_count=6).to(device)
    discriminator = ConditionalPatchDiscriminator(base_channels=8).to(device)
    image = torch.randn(2, 3, 256, 256, device=device)
    mask = torch.zeros(2, 1, 256, 256, device=device)
    mask[:, :, 48:112, 72:136] = 1.0
    masked = image * (1.0 - mask) + mask

    restored = generator(masked, mask)
    logits = discriminator(restored, mask)
    assert restored.shape == image.shape
    assert logits.ndim == 4 and logits.shape[:2] == (2, 1)
    assert torch.isfinite(restored).all()
    assert restored.min() >= -1.0 and restored.max() <= 1.0

    (restored.abs().mean() + logits.abs().mean()).backward()
    assert any(parameter.grad is not None for parameter in generator.parameters())
    assert any(parameter.grad is not None for parameter in discriminator.parameters())
    print(f"HFI-GAN smoke test passed on {device}: {tuple(restored.shape)}, {tuple(logits.shape)}")


if __name__ == "__main__":
    main()