"""Real-time processing pipeline orchestrator."""

from typing import List
import numpy as np
from .types import Detection
from .config import Config


class RealtimePipeline:
    """Orchestrates frame-by-frame processing."""
    
    def __init__(self, config: Config):
        """Initialize processing pipeline."""
        self.config = config
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Process single frame through pipeline."""
        # Placeholder for pipeline processing
        pass
    
    def should_run_detection(self, frame_idx: int) -> bool:
        """Determine if detection should run on this frame."""
        # Placeholder for detection scheduling
        pass
    
    def update_tracks(self, detections: List[Detection]) -> None:
        """Update tracking state with new detections."""
        # Placeholder for track updates
        pass