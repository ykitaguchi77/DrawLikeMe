"""
Image preprocessors for structure extraction.

Medical illustrations require high-fidelity structure extraction to preserve
anatomical accuracy. Multiple extraction methods are provided so users can
choose the one that best captures the structural detail of their specific
illustration type.
"""

import cv2
import numpy as np
from PIL import Image


def extract_canny(
    image: Image.Image,
    low_threshold: int = 80,
    high_threshold: int = 200,
) -> Image.Image:
    """
    Extract Canny edges from an image.

    Good for: sharp, well-defined edges in medical diagrams and illustrations.
    Lower threshold captures finer anatomical detail.
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return Image.fromarray(edges).convert("RGB")


def extract_lineart(
    image: Image.Image,
    gaussian_sigma: float = 4.0,
    intensity_threshold: int = 6,
) -> Image.Image:
    """
    Extract lineart using Difference of Gaussians (DoG).

    Lightweight method that doesn't require model downloads.
    Tuned for medical illustrations: finer sigma captures more anatomical detail.
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float64)

    # Difference of Gaussians for edge detection
    blur1 = cv2.GaussianBlur(gray, (0, 0), gaussian_sigma)
    blur2 = cv2.GaussianBlur(gray, (0, 0), gaussian_sigma * 1.6)
    dog = blur1 - blur2

    # Normalize and threshold
    dog = np.clip(dog, 0, 255)
    if dog.max() > 0:
        dog = (dog / dog.max() * 255).astype(np.uint8)
    else:
        dog = dog.astype(np.uint8)

    # Invert so lines are dark on white (illustration-style)
    lineart = 255 - dog
    lineart[lineart > (255 - intensity_threshold)] = 255

    return Image.fromarray(lineart).convert("RGB")


def extract_adaptive_threshold(
    image: Image.Image,
    block_size: int = 11,
    constant: int = 2,
) -> Image.Image:
    """
    Extract structure using adaptive thresholding.

    Good for: medical illustrations with varying contrast across regions
    (e.g., cross-sectional anatomy where different tissue types have
    different contrast levels).
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Apply bilateral filter first to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, constant,
    )

    return Image.fromarray(thresh).convert("RGB")


def extract_lineart_model(image: Image.Image) -> Image.Image:
    """
    Extract lineart using the controlnet_aux LineartDetector neural network.

    Highest quality lineart extraction. Downloads model weights (~400MB)
    on first call. Recommended for production use with medical illustrations.
    """
    from controlnet_aux import LineartDetector

    detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
    return detector(image)
