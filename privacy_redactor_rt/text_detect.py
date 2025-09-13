"""Text detection using PaddleOCR."""

from typing import List
import numpy as np
from .types import BBox
from .config import DetectionConfig


class TextDetector:
    """PaddleOCR-based text detection with lazy initialization."""
    
    def __init__(self, config: DetectionConfig):
        """Initialize text detector."""
        self.config = config
        self._detector = None
    
    def lazy_init(self) -> None:
        """Initialize PaddleOCR model on first use."""
        # Placeholder for lazy initialization
        pass
    
    def detect(self, frame: np.ndarray) -> List[BBox]:
        """Detect text regions in frame."""
        # Placeholder for text detection
        pass