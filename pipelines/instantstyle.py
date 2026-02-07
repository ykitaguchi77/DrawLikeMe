"""
Method B: InstantStyle + ControlNet Pipeline  [SDXL]

Structure preservation: Canny ControlNet
Style injection:        InstantStyle (IP-Adapter injected into style-specific
                        attention blocks only)

InstantStyle improves on plain IP-Adapter by:
1. Injecting style features ONLY into style-relevant attention blocks
   (up_blocks in SDXL), reducing content leakage from the style reference
2. This keeps anatomical structures intact while transferring line quality,
   coloring, and rendering style

Reference: https://github.com/instantX-research/InstantStyle
"""

import torch
from PIL import Image

from config import (
    SD_MODEL_ID,
    CONTROLNET_CANNY_MODEL,
    IP_ADAPTER_MODEL,
    IP_ADAPTER_SDXL_WEIGHT,
    IP_ADAPTER_IMAGE_ENCODER,
    DEFAULT_CONTROLNET_SCALE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_IP_ADAPTER_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_PROMPT,
    DEFAULT_NEGATIVE_PROMPT,
)
from pipelines.base import BasePipeline, StyleTransferResult
from pipelines.preprocessors import extract_canny
from utils.image_utils import prepare_image


class InstantStylePipeline(BasePipeline):
    """
    InstantStyle: ControlNet (structure) + style-only IP-Adapter injection.

    Style features are injected only into the style-specific cross-attention
    layers (up_blocks in SDXL), preventing the style reference from overriding
    anatomical content/structure. This makes it well-suited for medical
    illustrations where structure accuracy is paramount.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._pipe = None

    @property
    def name(self) -> str:
        return "InstantStyle (SDXL)"

    @property
    def description(self) -> str:
        return (
            "Improved IP-Adapter that injects style features ONLY into "
            "style-specific attention blocks. Better separation of content "
            "and style than Method A, reducing risk of anatomical distortion."
        )

    def _load_pipeline(self):
        if self._pipe is not None:
            return

        from diffusers import (
            StableDiffusionXLControlNetPipeline,
            ControlNetModel,
            AutoencoderKL,
        )
        from transformers import CLIPVisionModelWithProjection

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            IP_ADAPTER_IMAGE_ENCODER,
            torch_dtype=torch.float16,
        )

        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_CANNY_MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
        )

        self._pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SD_MODEL_ID,
            controlnet=controlnet,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        self._pipe.load_ip_adapter(
            IP_ADAPTER_MODEL,
            subfolder="sdxl_models",
            weight_name=IP_ADAPTER_SDXL_WEIGHT,
        )

        self._pipe.enable_model_cpu_offload()

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

        content = prepare_image(content_image)
        style = prepare_image(style_image)

        canny_image = extract_canny(content)

        # InstantStyle: apply scale ONLY to style-related blocks
        # In SDXL, up_blocks.0.attentions.1 is the primary style block.
        # A higher scale is acceptable here because it only affects style blocks.
        self._pipe.set_ip_adapter_scale({
            "up_blocks.0.attentions.1": style_strength,
        })

        if not prompt:
            prompt = DEFAULT_PROMPT
        if not negative_prompt:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canny_image,
            ip_adapter_image=style,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )

        return StyleTransferResult(
            output_image=result.images[0],
            method_name=self.name,
            parameters={
                "controlnet_scale": controlnet_scale,
                "style_strength": style_strength,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "base_model": "SDXL",
                "structure_extraction": "Canny",
                "style_injection": "up_blocks.0.attentions.1 only",
            },
            preprocessing_images={"canny_edges": canny_image},
        )
