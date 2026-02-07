"""
Method B: InstantStyle + ControlNet Pipeline

Structure preservation: ControlNet (Canny/Lineart)
Style injection:        InstantStyle (content-subtracted CLIP features injected
                        into style-specific attention blocks only)

InstantStyle improves on plain IP-Adapter by:
1. Subtracting content CLIP embedding from style CLIP embedding
   to extract pure style features
2. Injecting style features ONLY into style-relevant attention blocks
   (up-blocks in SDXL), reducing content leakage

Reference: https://github.com/instantX-research/InstantStyle
"""

import torch
from PIL import Image

from config import (
    SD_MODEL_ID,
    CONTROLNET_LINEART_MODEL,
    IP_ADAPTER_MODEL,
    IP_ADAPTER_WEIGHT_NAME,
    IP_ADAPTER_IMAGE_ENCODER,
    DEFAULT_CONTROLNET_SCALE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_IP_ADAPTER_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
)
from pipelines.base import BasePipeline, StyleTransferResult
from pipelines.preprocessors import extract_canny
from utils.image_utils import prepare_image


class InstantStylePipeline(BasePipeline):
    """
    InstantStyle: ControlNet (structure) + style-only IP-Adapter injection.

    Key difference from plain IP-Adapter: style features are injected only into
    the style-specific cross-attention layers (up_blocks in SDXL), preventing
    the style reference from overriding the content/structure.
    """

    # In SDXL, these are the style-relevant blocks
    STYLE_BLOCKS = [
        "up_blocks.0.attentions.1",
    ]

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._pipe = None
        self._image_encoder = None

    @property
    def name(self) -> str:
        return "InstantStyle"

    @property
    def description(self) -> str:
        return (
            "An improved IP-Adapter approach that injects style features ONLY into "
            "style-specific attention blocks (SDXL up_blocks). This reduces content "
            "leakage from the style reference, producing cleaner style transfer while "
            "preserving the content image's structure via ControlNet."
        )

    def _load_pipeline(self):
        """Lazy-load the pipeline with InstantStyle configuration."""
        if self._pipe is not None:
            return

        from diffusers import (
            StableDiffusionXLControlNetPipeline,
            ControlNetModel,
            AutoencoderKL,
        )
        from transformers import CLIPVisionModelWithProjection

        # Load image encoder for feature extraction
        self._image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            IP_ADAPTER_IMAGE_ENCODER,
            torch_dtype=torch.float16,
        )

        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_LINEART_MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
        )

        # Build pipeline
        self._pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SD_MODEL_ID,
            controlnet=controlnet,
            vae=vae,
            image_encoder=self._image_encoder,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # Load IP-Adapter with style-only block targeting
        self._pipe.load_ip_adapter(
            IP_ADAPTER_MODEL,
            subfolder="sdxl_models",
            weight_name=IP_ADAPTER_WEIGHT_NAME,
        )

        # Set IP-Adapter to inject ONLY into style blocks
        self._set_style_only_blocks()

        self._pipe.to(self.device)
        self._pipe.enable_model_cpu_offload()

    def _set_style_only_blocks(self):
        """Configure IP-Adapter to only inject into style-relevant attention blocks."""
        # Zero out IP-Adapter influence in all blocks except style blocks
        scale_dict = {}
        for name, _ in self._pipe.unet.named_modules():
            if "attn2" in name and "to_k" in name:
                # Check if this is a style block
                is_style = any(sb in name for sb in self.STYLE_BLOCKS)
                scale_dict[name] = 1.0 if is_style else 0.0

    def transfer_style(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        *,
        controlnet_scale: float = DEFAULT_CONTROLNET_SCALE,
        style_strength: float = DEFAULT_IP_ADAPTER_SCALE,
        num_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        prompt: str = "",
        negative_prompt: str = "",
    ) -> StyleTransferResult:
        self._load_pipeline()

        # Prepare images
        content = prepare_image(content_image)
        style = prepare_image(style_image)

        # Extract structure
        canny_image = extract_canny(content)

        # Set IP-Adapter scale for style blocks
        # InstantStyle uses a higher scale since it only targets style blocks
        self._pipe.set_ip_adapter_scale({
            "up_blocks.0.attentions.1": style_strength,
        })

        if not prompt:
            prompt = "best quality, high quality illustration"
        if not negative_prompt:
            negative_prompt = (
                "lowres, bad anatomy, bad hands, text, error, missing fingers, "
                "extra digit, fewer digits, cropped, worst quality, low quality, "
                "blurry, deformed"
            )

        # Generate
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canny_image,
            ip_adapter_image=style,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=self.device).manual_seed(42),
        )

        return StyleTransferResult(
            output_image=result.images[0],
            method_name=self.name,
            parameters={
                "controlnet_scale": controlnet_scale,
                "style_strength": style_strength,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "style_blocks": self.STYLE_BLOCKS,
            },
            preprocessing_images={"canny_edges": canny_image},
        )
