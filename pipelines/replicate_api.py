"""
Method C: Replicate API-based Style Transfer

Uses cloud GPU inference via Replicate's hosted models.
No local GPU required. Supports multiple model backends.

Available models:
  - fofr/style-transfer: ControlNet + IP-Adapter (recommended)
  - Other Replicate models can be added as needed
"""

import io
import time
import requests
from PIL import Image

from config import REPLICATE_API_TOKEN, REPLICATE_STYLE_TRANSFER_MODEL
from pipelines.base import BasePipeline, StyleTransferResult
from utils.image_utils import prepare_image, pil_to_data_uri


class ReplicateStyleTransferPipeline(BasePipeline):
    """Style transfer via Replicate API (cloud GPU inference)."""

    def __init__(self):
        if not REPLICATE_API_TOKEN:
            self._replicate = None
        else:
            import replicate
            self._replicate = replicate

    @property
    def name(self) -> str:
        return "Replicate API (Cloud)"

    @property
    def description(self) -> str:
        return (
            "Cloud-based style transfer using Replicate's hosted models. "
            "Uses fofr/style-transfer (ControlNet + IP-Adapter internally). "
            "No local GPU required. Requires REPLICATE_API_TOKEN env var. "
            "Cost: ~$0.02-0.08 per image."
        )

    @property
    def is_available(self) -> bool:
        return bool(REPLICATE_API_TOKEN)

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
        if not self.is_available:
            raise RuntimeError(
                "Replicate API token not set. "
                "Set the REPLICATE_API_TOKEN environment variable."
            )

        import replicate

        # Prepare images
        content = prepare_image(content_image)
        style = prepare_image(style_image)

        # Convert to data URIs for Replicate API
        content_uri = pil_to_data_uri(content, "PNG")
        style_uri = pil_to_data_uri(style, "PNG")

        if not prompt:
            prompt = "best quality, high quality illustration, in the style of the reference"

        # Run model
        output = replicate.run(
            REPLICATE_STYLE_TRANSFER_MODEL,
            input={
                "image": content_uri,
                "style_image": style_uri,
                "prompt": prompt,
                "negative_prompt": negative_prompt or "lowres, bad quality, blurry",
                "control_depth_strength": controlnet_scale,
                "ip_adapter_noise": 0.5,
                "ip_adapter_weight": style_strength,
                "num_inference_steps": num_steps,
                "guidance_scale": guidance_scale,
            },
        )

        # Download result
        output_url = output[0] if isinstance(output, list) else str(output)
        response = requests.get(output_url, timeout=60)
        response.raise_for_status()
        result_image = Image.open(io.BytesIO(response.content)).convert("RGB")

        return StyleTransferResult(
            output_image=result_image,
            method_name=self.name,
            parameters={
                "model": REPLICATE_STYLE_TRANSFER_MODEL,
                "controlnet_scale": controlnet_scale,
                "style_strength": style_strength,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
            },
        )
