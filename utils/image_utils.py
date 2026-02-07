"""Image utility functions."""

import io
import base64
from PIL import Image

from config import IMAGE_MAX_SIZE


def resize_to_max(image: Image.Image, max_size: int = IMAGE_MAX_SIZE) -> Image.Image:
    """Resize image so that the longest side is at most max_size, preserving aspect ratio."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    # Round to nearest multiple of 8 (required by many diffusion models)
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8
    return image.resize((new_w, new_h), Image.LANCZOS)


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB if necessary."""
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_data_uri(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to data URI."""
    b64 = pil_to_base64(image, fmt)
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def prepare_image(image: Image.Image, max_size: int = IMAGE_MAX_SIZE) -> Image.Image:
    """Standard preparation: ensure RGB and resize."""
    return resize_to_max(ensure_rgb(image), max_size)
