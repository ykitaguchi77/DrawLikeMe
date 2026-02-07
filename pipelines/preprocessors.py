"""Image preprocessors for structure extraction (Canny, Lineart, etc.)."""

import cv2
import numpy as np
from PIL import Image


def extract_canny(
    image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Image.Image:
    """
    Extract Canny edges from an image.

    Args:
        image: Input PIL Image (RGB).
        low_threshold: Lower hysteresis threshold.
        high_threshold: Upper hysteresis threshold.

    Returns:
        PIL Image of Canny edges (white edges on black background).
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return Image.fromarray(edges).convert("RGB")


def extract_lineart(
    image: Image.Image,
    gaussian_sigma: float = 6.0,
    intensity_threshold: int = 8,
) -> Image.Image:
    """
    Extract lineart from an image using Difference of Gaussians (DoG).

    This is a lightweight alternative to the full controlnet_aux LineartDetector
    that doesn't require downloading a model. For production use, prefer the
    model-based detector from controlnet_aux.

    Args:
        image: Input PIL Image (RGB).
        gaussian_sigma: Sigma for Gaussian blur.
        intensity_threshold: Minimum intensity for edge pixels.

    Returns:
        PIL Image of extracted lineart (dark lines on white background).
    """
    img_array = np.array(image).astype(np.float64)
    gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)

    # Difference of Gaussians for edge detection
    blur1 = cv2.GaussianBlur(gray, (0, 0), gaussian_sigma)
    blur2 = cv2.GaussianBlur(gray, (0, 0), gaussian_sigma * 1.6)
    dog = blur1 - blur2

    # Normalize and threshold
    dog = np.clip(dog, 0, 255)
    dog = (dog / dog.max() * 255).astype(np.uint8) if dog.max() > 0 else dog.astype(np.uint8)

    # Invert so lines are dark on white (illustration-style)
    lineart = 255 - dog
    lineart[lineart > (255 - intensity_threshold)] = 255

    return Image.fromarray(lineart).convert("RGB")


def extract_lineart_model(image: Image.Image) -> Image.Image:
    """
    Extract lineart using the controlnet_aux LineartDetector model.

    This gives higher quality results than the DoG-based method but requires
    downloading the model weights (~400MB).

    Args:
        image: Input PIL Image (RGB).

    Returns:
        PIL Image of extracted lineart.
    """
    from controlnet_aux import LineartDetector

    detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
    return detector(image)


def extract_lineart_anime(image: Image.Image) -> Image.Image:
    """
    Extract anime-style lineart using the controlnet_aux LineartAnimeDetector.

    Best for anime/manga-style illustrations.

    Args:
        image: Input PIL Image (RGB).

    Returns:
        PIL Image of extracted anime lineart.
    """
    from controlnet_aux import LineartAnimeDetector

    detector = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    return detector(image)
