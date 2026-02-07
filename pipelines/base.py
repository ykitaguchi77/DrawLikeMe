"""Abstract base class for style transfer pipelines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image


@dataclass
class StyleTransferResult:
    """Result of a style transfer operation."""
    output_image: Image.Image
    method_name: str
    parameters: dict
    preprocessing_images: dict[str, Image.Image] | None = None


class BasePipeline(ABC):
    """Base class for all style transfer pipelines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this pipeline."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of how this pipeline works."""
        ...

    @abstractmethod
    def transfer_style(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        *,
        controlnet_scale: float = 0.8,
        style_strength: float = 0.6,
        num_steps: int = 30,
        guidance_scale: float = 7.5,
        prompt: str = "",
        negative_prompt: str = "",
    ) -> StyleTransferResult:
        """
        Transfer style from style_image onto content_image.

        Args:
            content_image: The image whose structure/composition to preserve.
            style_image:   The reference illustration whose style to apply.
            controlnet_scale: How strongly to enforce structural constraints (0-1).
            style_strength:   How strongly to apply the style (0-1).
            num_steps:        Number of denoising steps.
            guidance_scale:   Classifier-free guidance scale.
            prompt:           Optional text prompt to guide generation.
            negative_prompt:  Optional negative prompt.

        Returns:
            StyleTransferResult with the output image and metadata.
        """
        ...
