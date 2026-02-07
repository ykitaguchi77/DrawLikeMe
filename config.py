"""DrawLikeMe - Configuration"""

import os

# --- Device ---
# "auto" will detect GPU availability, or force "cpu" / "cuda"
DEVICE = os.environ.get("DRAWLIKEME_DEVICE", "auto")

# --- Model Settings ---

# Base model for local pipelines
SD_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# ControlNet models (Canny for SDXL)
CONTROLNET_CANNY_MODEL = "diffusers/controlnet-canny-sdxl-1.0"

# Lineart ControlNet (SD 1.5 based - more mature for lineart)
CONTROLNET_LINEART_SD15_MODEL = "lllyasviel/control_v11p_sd15_lineart"
SD15_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# IP-Adapter
IP_ADAPTER_MODEL = "h94/IP-Adapter"
IP_ADAPTER_SDXL_WEIGHT = "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
IP_ADAPTER_SD15_WEIGHT = "models/ip-adapter-plus_sd15.safetensors"
IP_ADAPTER_IMAGE_ENCODER = "h94/IP-Adapter/models/image_encoder"
IP_ADAPTER_SD15_IMAGE_ENCODER = "h94/IP-Adapter/sdlight_models/image_encoder"

# --- Generation Defaults (medical illustration optimized) ---
# Higher controlnet scale for anatomical accuracy
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_CONTROLNET_SCALE = 0.9   # High: medical illustrations need precise structure
DEFAULT_IP_ADAPTER_SCALE = 0.5   # Moderate: apply style without distorting anatomy
DEFAULT_STRENGTH = 0.75
IMAGE_MAX_SIZE = 1024

# --- Medical Illustration Prompts ---
DEFAULT_PROMPT = (
    "best quality, high quality, detailed medical illustration, "
    "clean lines, professional illustration, accurate anatomy"
)
DEFAULT_NEGATIVE_PROMPT = (
    "lowres, blurry, deformed, distorted anatomy, incorrect anatomy, "
    "bad proportions, extra limbs, missing parts, text, watermark, "
    "worst quality, low quality, artifacts, noise"
)
