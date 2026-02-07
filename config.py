"""DrawLikeMe - Configuration"""

import os

# --- API Keys ---
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")

# --- Device ---
# "auto" will detect GPU availability, or force "cpu" / "cuda"
DEVICE = os.environ.get("DRAWLIKEME_DEVICE", "auto")

# --- Model Settings ---

# Base model for local pipelines
SD_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# ControlNet models
CONTROLNET_LINEART_MODEL = "diffusers/controlnet-canny-sdxl-1.0"

# IP-Adapter
IP_ADAPTER_MODEL = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT_NAME = "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
IP_ADAPTER_IMAGE_ENCODER = "h94/IP-Adapter/models/image_encoder"

# Replicate models
REPLICATE_STYLE_TRANSFER_MODEL = "fofr/style-transfer"
REPLICATE_FLUX_CANNY_MODEL = "black-forest-labs/flux-1.1-pro-ultra"

# --- Generation Defaults ---
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_CONTROLNET_SCALE = 0.8
DEFAULT_IP_ADAPTER_SCALE = 0.6
DEFAULT_STRENGTH = 0.75
IMAGE_MAX_SIZE = 1024
