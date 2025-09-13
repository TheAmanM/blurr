"""Video input normalization and frame processing utilities.

This module provides frame normalization to 1280×720 with letterboxing,
FPS throttling, and support for multiple input formats (webcam, RTSP, file).
"""

import time
from typing import Optional, Tuple, Union, Iterator
from pathlib import Path
import cv2
import numpy as np
from .config import IOConfig
from .types import BBox


class FrameNormalizer:
    """Handles frame normalization with letterboxing and aspect ratio preservation."""
    
    def __init__(self, config: IOConfig):
        """Initialize frame normalizer with configuration.
        
        Args:
            config: IO configuration containing target dimensions and letterbox settings
        """
        self.config = config
        self.target_width = config.target_width
        self.target_height = config.target_height
        self.letterbox = config.letterbox
        self.letterbox_color = config.letterbox_color
        
    def normalize_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Normalize frame to target dimensions with letterboxing.
        
        Args:
            frame: Input frame as BGR numpy array
            
        Returns:
            Tuple of (normalized_frame, scale_factor, (offset_x, offset_y))
            - normalized_frame: Frame resized to target dimensions
            - scale_factor: Scaling factor applied to original frame
            - offset_x, offset_y: Letterbox offsets for coordinate mapping
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame")
            
        original_height, original_width = frame.shape[:2]
        
        if not self.letterbox:
            # Simple resize without aspect ratio preservation
            resized = cv2.resize(frame, (self.target_width, self.target_height))
            scale_x = self.target_width / original_width
            scale_y = self.target_height / original_height
            return resized, min(scale_x, scale_y), (0, 0)
        
        # Calculate scaling to fit within target dimensions while preserving aspect ratio
        scale_x = self.target_width / original_width
        scale_y = self.target_height / original_height
        scale = min(scale_x, scale_y)
        
        # Calculate new dimensions after scaling
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize frame maintaining aspect ratio
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Create letterboxed frame with target dimensions
        letterboxed = np.full(
            (self.target_height, self.target_width, 3),
            self.letterbox_color,
            dtype=np.uint8
        )
        
        # Calculate offsets to center the resized frame
        offset_x = (self.target_width - new_width) // 2
        offset_y = (self.target_height - new_height) // 2
        
        # Place resized frame in center of letterboxed frame
        letterboxed[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized
        
        return letterboxed, scale, (offset_x, offset_y)
    
    def denormalize_bbox(self, bbox: BBox, scale: float, offset: Tuple[int, int]) -> BBox:
        """Convert bounding box from normalized coordinates back to original frame coordinates.
        
        Args:
            bbox: Bounding box in normalized frame coordinates
            scale: Scale factor used during normalization
            offset: Letterbox offset (x, y)
            
        Returns:
            BBox in original frame coordinates
        """
        offset_x, offset_y = offset
        
        # Remove letterbox offset
        x1 = bbox.x1 - offset_x
        y1 = bbox.y1 - offset_y
        x2 = bbox.x2 - offset_x
        y2 = bbox.y2 - offset_y
        
        # Scale back to original coordinates
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)
        
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=bbox.confidence)
    
    def normalize_bbox(self, bbox: BBox, scale: float, offset: Tuple[int, int]) -> BBox:
        """Convert bounding box from original coordinates to normalized frame coordinates.
        
        Args:
            bbox: Bounding box in original frame coordinates
            scale: Scale factor used during normalization
            offset: Letterbox offset (x, y)
            
        Returns:
            BBox in normalized frame coordinates
        """
        offset_x, offset_y = offset
        
        # Scale to normalized coordinates
        x1 = int(bbox.x1 * scale)
        y1 = int(bbox.y1 * scale)
        x2 = int(bbox.x2 * scale)
        y2 = int(bbox.y2 * scale)
        
        # Add letterbox offset
        x1 += offset_x
        y1 += offset_y
        x2 += offset_x
        y2 += offset_y
        
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=bbox.confidence)


class FPSThrottler:
    """Maintains consistent FPS output by throttling frame processing."""
    
    def __init__(self, target_fps: int):
        """Initialize FPS throttler.
        
        Args:
            target_fps: Target frames per second
        """
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_frame_time = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        
    def should_process_frame(self) -> bool:
        """Check if enough time has passed to process the next frame.
        
        Returns:
            True if frame should be processed, False to skip
        """
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        if elapsed >= self.frame_interval:
            self.last_frame_time = current_time
            self.frame_count += 1
            return True
        return False
    
    def wait_for_next_frame(self) -> None:
        """Wait until it's time for the next frame to maintain target FPS."""
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        if elapsed < self.frame_interval:
            sleep_time = self.frame_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_frame_time = time.time()
        self.frame_count += 1
    
    def get_actual_fps(self) -> float:
        """Calculate actual FPS based on processed frames.
        
        Returns:
            Actual frames per second
        """
        if self.frame_count == 0:
            return 0.0
        
        elapsed_total = time.time() - self.start_time
        return self.frame_count / elapsed_total if elapsed_total > 0 else 0.0
    
    def reset(self) -> None:
        """Reset FPS tracking statistics."""
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = 0.0


class VideoSource:
    """Unified video source interface supporting webcam, RTSP, and file inputs."""
    
    def __init__(self, config: IOConfig):
        """Initialize video source with configuration.
        
        Args:
            config: IO configuration
        """
        self.config = config
        self.normalizer = FrameNormalizer(config)
        self.fps_throttler = FPSThrottler(config.target_fps)
        self.cap: Optional[cv2.VideoCapture] = None
        self.source_fps: Optional[float] = None
        self.total_frames: Optional[int] = None
        self.current_frame: int = 0
        
    def open_webcam(self, device_id: int = 0) -> bool:
        """Open webcam as video source.
        
        Args:
            device_id: Webcam device ID (default: 0)
            
        Returns:
            True if successfully opened, False otherwise
        """
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            return False
            
        # Set webcam properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual webcam FPS
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.source_fps <= 0:
            self.source_fps = 30.0  # Default fallback
            
        self.total_frames = None  # Infinite for webcam
        self.current_frame = 0
        return True
    
    def open_rtsp(self, url: str) -> bool:
        """Open RTSP/HTTP stream as video source.
        
        Args:
            url: RTSP or HTTP URL
            
        Returns:
            True if successfully opened, False otherwise
        """
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            return False
            
        # Configure for network streams
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
        
        # Get stream FPS
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.source_fps <= 0:
            self.source_fps = 30.0  # Default fallback
            
        self.total_frames = None  # Unknown for streams
        self.current_frame = 0
        return True
    
    def open_file(self, file_path: Union[str, Path]) -> bool:
        """Open video file as source.
        
        Args:
            file_path: Path to video file
            
        Returns:
            True if successfully opened, False otherwise
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return False
            
        self.cap = cv2.VideoCapture(str(file_path))
        if not self.cap.isOpened():
            return False
            
        # Get file properties
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        
        if self.source_fps <= 0:
            self.source_fps = 30.0  # Default fallback
            
        return True
    
    def read_frame(self) -> Optional[Tuple[np.ndarray, float, Tuple[int, int]]]:
        """Read and normalize next frame from video source.
        
        Returns:
            Tuple of (normalized_frame, scale_factor, offset) or None if no frame available
        """
        if self.cap is None or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
            
        self.current_frame += 1
        
        # Normalize frame
        try:
            normalized_frame, scale, offset = self.normalizer.normalize_frame(frame)
            return normalized_frame, scale, offset
        except ValueError:
            return None
    
    def read_frame_throttled(self) -> Optional[Tuple[np.ndarray, float, Tuple[int, int]]]:
        """Read frame with FPS throttling for file sources.
        
        Returns:
            Tuple of (normalized_frame, scale_factor, offset) or None if no frame available
        """
        # For live sources (webcam, RTSP), don't throttle
        if self.total_frames is None:
            return self.read_frame()
        
        # For file sources, throttle to target FPS
        if not self.fps_throttler.should_process_frame():
            return None
            
        return self.read_frame()
    
    def get_frame_iterator(self, throttled: bool = True) -> Iterator[Tuple[np.ndarray, float, Tuple[int, int]]]:
        """Get iterator over video frames.
        
        Args:
            throttled: Whether to apply FPS throttling for file sources
            
        Yields:
            Tuples of (normalized_frame, scale_factor, offset)
        """
        while True:
            if throttled:
                result = self.read_frame_throttled()
            else:
                result = self.read_frame()
                
            if result is None:
                break
                
            yield result
    
    def get_progress(self) -> Optional[float]:
        """Get playback progress for file sources.
        
        Returns:
            Progress as fraction (0.0 to 1.0) or None for live sources
        """
        if self.total_frames is None or self.total_frames == 0:
            return None
        return min(1.0, self.current_frame / self.total_frames)
    
    def get_source_info(self) -> dict:
        """Get information about the video source.
        
        Returns:
            Dictionary with source information
        """
        if self.cap is None:
            return {}
            
        return {
            "source_fps": self.source_fps,
            "total_frames": self.total_frames,
            "current_frame": self.current_frame,
            "progress": self.get_progress(),
            "target_fps": self.config.target_fps,
            "target_resolution": (self.config.target_width, self.config.target_height),
            "actual_fps": self.fps_throttler.get_actual_fps()
        }
    
    def close(self) -> None:
        """Close video source and release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.fps_throttler.reset()
        self.current_frame = 0


def convert_bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to RGB.
    
    Args:
        frame: BGR frame
        
    Returns:
        RGB frame
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def convert_rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert RGB frame to BGR.
    
    Args:
        frame: RGB frame
        
    Returns:
        BGR frame
    """
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def preserve_aspect_ratio(
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int
) -> Tuple[int, int, float]:
    """Calculate dimensions that preserve aspect ratio within target bounds.
    
    Args:
        original_width: Original frame width
        original_height: Original frame height
        target_width: Target width constraint
        target_height: Target height constraint
        
    Returns:
        Tuple of (new_width, new_height, scale_factor)
    """
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale = min(scale_x, scale_y)
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    return new_width, new_height, scale