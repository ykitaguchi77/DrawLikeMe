"""
Method C: Lineart ControlNet + IP-Adapter Pipeline  [SD 1.5]

Structure preservation: Lineart ControlNet (purpose-built for line drawings)
Style injection:        IP-Adapter Plus (CLIP image features)

SD 1.5 has the most mature Lineart ControlNet model
(lllyasviel/control_v11p_sd15_lineart), which was specifically trained
to understand and preserve line art structure - ideal for medical
illustrations that rely on clean, precise lines.

Trade-off vs SDXL: lower resolution (512px) but better lineart fidelity.
"""

import torch
from PIL import Image

from config import (
    SD15_MODEL_ID,
    CONTROLNET_LINEART_SD15_MODEL,
    IP_ADAPTER_MODEL,
    IP_ADAPTER_SD15_WEIGHT,
    IP_ADAPTER_SD15_IMAGE_ENCODER,
    DEFAULT_CONTROLNET_SCALE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_IP_ADAPTER_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_PROMPT,
    DEFAULT_NEGATIVE_PROMPT,
)
from pipelines.base import BasePipeline, StyleTransferResult
from pipelines.preprocessors import extract_lineart
from utils.image_utils import prepare_image


class LineartIPAdapterPipeline(BasePipeline):
    """Lineart ControlNet (structure) + IP-Adapter (style) on SD 1.5."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._pipe = None

    @property
    def name(self) -> str:
        return "Lineart ControlNet + IP-Adapter (SD 1.5)"

    @property
    def description(self) -> str:
        return (
            "Uses a Lineart-specialized ControlNet trained specifically for "
            "line drawing preservation. Better at capturing fine lines and "
            "structural details than Canny. Lower resolution (512px) than SDXL "
            "methods but more precise lineart fidelity."
        )

    def _load_pipeline(self):
        if self._pipe is not None:
            return

        from diffusers import (
            StableDiffusionControlNetPipeline,
            ControlNetModel,
        )

        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_LINEART_SD15_MODEL,
            torch_dtype=torch.float16,
        )

        self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
            SD15_MODEL_ID,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )

        self._pipe.load_ip_adapter(
            IP_ADAPTER_MODEL,
            subfolder="models",
            weight_name=IP_ADAPTER_SD15_WEIGHT,
            image_encoder_folder=IP_ADAPTER_SD15_IMAGE_ENCODER,
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

        # SD 1.5 works best at 512px
        content = prepare_image(content_image, max_size=512)
        style = prepare_image(style_image, max_size=512)

        # Extract lineart structure (DoG-based, lightweight)
        lineart_image = extract_lineart(content)

        self._pipe.set_ip_adapter_scale(style_strength)

        if not prompt:
            prompt = DEFAULT_PROMPT
        if not negative_prompt:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=lineart_image,
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
                "ip_adapter_scale": style_strength,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "base_model": "SD 1.5",
                "structure_extraction": "Lineart (DoG)",
            },
            preprocessing_images={"lineart": lineart_image},
        )
