"""
DrawLikeMe pipelines.

Heavy pipeline classes are imported lazily to avoid loading torch/diffusers
when only preprocessors are needed (e.g., CPU-only preview mode).
"""

from pipelines.base import BasePipeline, StyleTransferResult
from pipelines.preprocessors import (
    extract_canny,
    extract_lineart,
    extract_adaptive_threshold,
)


def __getattr__(name: str):
    """Lazy import for GPU-dependent pipeline classes."""
    if name == "ControlNetIPAdapterPipeline":
        from pipelines.controlnet_ipadapter import ControlNetIPAdapterPipeline
        return ControlNetIPAdapterPipeline
    if name == "InstantStylePipeline":
        from pipelines.instantstyle import InstantStylePipeline
        return InstantStylePipeline
    if name == "LineartIPAdapterPipeline":
        from pipelines.lineart_ipadapter import LineartIPAdapterPipeline
        return LineartIPAdapterPipeline
    raise AttributeError(f"module 'pipelines' has no attribute {name!r}")


__all__ = [
    "BasePipeline",
    "StyleTransferResult",
    "extract_canny",
    "extract_lineart",
    "extract_adaptive_threshold",
    "ControlNetIPAdapterPipeline",
    "InstantStylePipeline",
    "LineartIPAdapterPipeline",
]
