"""
Method A: ControlNet + IP-Adapter Pipeline

Structure preservation: ControlNet (Canny/Lineart)
Style injection:        IP-Adapter Plus (CLIP image features)

This is the most established and well-tested combination.
IP-Adapter works zero-shot with a single reference image.
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
from pipelines.preprocessors import extract_canny, extract_lineart
from utils.image_utils import prepare_image


class ControlNetIPAdapterPipeline(BasePipeline):
    """ControlNet (structure) + IP-Adapter (style) pipeline using SDXL."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._pipe = None

    @property
    def name(self) -> str:
        return "ControlNet + IP-Adapter"

    @property
    def description(self) -> str:
        return (
            "ControlNet extracts edge/lineart from the content image to preserve structure. "
            "IP-Adapter Plus injects CLIP image features from the style reference to apply "
            "the artistic style. Zero-shot (no training needed), works with 1 reference image."
        )

    def _load_pipeline(self):
        """Lazy-load the full pipeline (downloads models on first call)."""
        if self._pipe is not None:
            return

        from diffusers import (
            StableDiffusionXLControlNetPipeline,
            ControlNetModel,
            AutoencoderKL,
        )

        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_LINEART_MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # Load VAE (better quality)
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
        )

        # Build pipeline
        self._pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SD_MODEL_ID,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # Load IP-Adapter
        self._pipe.load_ip_adapter(
            IP_ADAPTER_MODEL,
            subfolder="sdxl_models",
            weight_name=IP_ADAPTER_WEIGHT_NAME,
            image_encoder_folder=IP_ADAPTER_IMAGE_ENCODER,
        )

        self._pipe.to(self.device)

        # Enable memory optimizations
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

        # Prepare images
        content = prepare_image(content_image)
        style = prepare_image(style_image)

        # Extract structure from content image
        canny_image = extract_canny(content)

        # Set IP-Adapter scale
        self._pipe.set_ip_adapter_scale(style_strength)

        # Default prompts
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
                "ip_adapter_scale": style_strength,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
            },
            preprocessing_images={"canny_edges": canny_image},
        )
