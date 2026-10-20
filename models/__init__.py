"""Original HFI-GAN model components."""

from .discriminator import ConditionalPatchDiscriminator
from .hybrid_generator import HybridForensicGenerator

__all__ = ["HybridForensicGenerator", "ConditionalPatchDiscriminator"]