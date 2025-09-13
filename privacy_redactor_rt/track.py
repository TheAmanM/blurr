"""Optical flow tracking with IoU association."""

from typing import List
import numpy as np
from .types import Detection, Track
from .config import TrackingConfig


class OpticalFlowTracker:
    """Sparse optical flow tracking with IoU association."""
    
    def __init__(self, config: TrackingConfig):
        """Initialize optical flow tracker."""
        self.config = config
        self._tracks: List[Track] = []
        self._next_id = 0
    
    def propagate_tracks(self, frame: np.ndarray, prev_frame: np.ndarray) -> None:
        """Propagate tracks using optical flow."""
        # Placeholder for optical flow propagation
        pass
    
    def associate_detections(self, detections: List[Detection]) -> None:
        """Associate detections with existing tracks."""
        # Placeholder for track association
        pass
    
    def get_active_tracks(self) -> List[Track]:
        """Get currently active tracks."""
        # Placeholder for track filtering
        pass