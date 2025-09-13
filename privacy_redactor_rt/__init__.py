"""Privacy Redactor RT - Real-time sensitive information detection and redaction system."""

__version__ = "0.1.0"
__author__ = "Privacy Redactor RT Team"
__description__ = "Real-time video processing pipeline for detecting and redacting sensitive information"

from .types import BBox, Detection, Match, Track
from .config import Config

__all__ = [
    "BBox",
    "Detection", 
    "Match",
    "Track",
    "Config",
]