"""Core data types for the privacy redactor system."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class BBox:
    """Bounding box with confidence score."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass
class Detection:
    """Text detection result with optional OCR text."""
    bbox: BBox
    text: Optional[str]
    timestamp: float


@dataclass
class Match:
    """Classification match for sensitive data."""
    category: str
    confidence: float
    masked_text: str
    bbox: BBox


@dataclass
class Track:
    """Temporal track for bounding box across frames."""
    id: str
    bbox: BBox
    matches: List[Match]
    age: int
    hits: int
    last_ocr_frame: int
    flow_points: np.ndarray