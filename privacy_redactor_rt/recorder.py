"""MP4 recording with audio preservation."""

import numpy as np
from .config import Config


class MP4Recorder:
    """FFmpeg-based MP4 recording with audio support."""
    
    def __init__(self, config: Config, output_path: str):
        """Initialize MP4 recorder."""
        self.config = config
        self.output_path = output_path
    
    def start_recording(self) -> None:
        """Start recording session."""
        # Placeholder for recording startup
        pass
    
    def write_frame(self, frame: np.ndarray) -> None:
        """Write frame to recording."""
        # Placeholder for frame writing
        pass
    
    def stop_recording(self) -> None:
        """Stop recording and finalize file."""
        # Placeholder for recording cleanup
        pass