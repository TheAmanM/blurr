"""Redaction engine with multiple methods for obscuring sensitive information."""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum

from .types import Track, BBox
from .config import RedactionConfig


class RedactionMethod(Enum):
    """Supported redaction methods."""
    GAUSSIAN = "gaussian"
    PIXELATE = "pixelate" 
    SOLID = "solid"


class RedactionEngine:
    """Engine for applying various redaction methods to sensitive regions."""
    
    def __init__(self, config: RedactionConfig):
        """Initialize redaction engine with configuration.
        
        Args:
            config: Redaction configuration parameters
        """
        self.config = config
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate redaction configuration parameters."""
        # Validate default method
        try:
            RedactionMethod(self.config.default_method)
        except ValueError:
            raise ValueError(f"Invalid default redaction method: {self.config.default_method}")
            
        # Validate category-specific methods
        for category, method in self.config.category_methods.items():
            try:
                RedactionMethod(method)
            except ValueError:
                raise ValueError(f"Invalid redaction method '{method}' for category '{category}'")
                
        # Validate gaussian parameters
        if self.config.gaussian_kernel_size % 2 == 0:
            raise ValueError("Gaussian kernel size must be odd")
            
        if self.config.gaussian_kernel_size < 3:
            raise ValueError("Gaussian kernel size must be at least 3")
            
        # Validate pixelation parameters
        if self.config.pixelate_block_size < 2:
            raise ValueError("Pixelate block size must be at least 2")
            
    def redact_regions(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Apply redaction to all tracked regions in the frame.
        
        Args:
            frame: Input frame as BGR numpy array
            tracks: List of active tracks with matches
            
        Returns:
            Frame with redacted regions
        """
        if not tracks:
            return frame
            
        # Work on a copy to avoid modifying the original
        redacted_frame = frame.copy()
        
        # Group tracks by redaction method for efficiency
        method_groups: Dict[RedactionMethod, List[Track]] = {}
        
        for track in tracks:
            # Skip tracks without matches
            if not track.matches:
                continue
                
            # Get the best match to determine category
            best_match = track.get_best_match()
            if not best_match:
                continue
                
            # Determine redaction method for this category
            method_name = self.config.category_methods.get(
                best_match.category, 
                self.config.default_method
            )
            method = RedactionMethod(method_name)
            
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(track)
            
        # Apply redaction for each method group
        for method, method_tracks in method_groups.items():
            redacted_frame = self._apply_method_to_tracks(
                redacted_frame, method_tracks, method
            )
            
        return redacted_frame
        
    def _apply_method_to_tracks(
        self, 
        frame: np.ndarray, 
        tracks: List[Track], 
        method: RedactionMethod
    ) -> np.ndarray:
        """Apply a specific redaction method to a group of tracks.
        
        Args:
            frame: Input frame
            tracks: Tracks to redact with the same method
            method: Redaction method to apply
            
        Returns:
            Frame with redaction applied
        """
        for track in tracks:
            frame = self._redact_single_region(frame, track.bbox, method)
        return frame
        
    def _redact_single_region(
        self, 
        frame: np.ndarray, 
        bbox: BBox, 
        method: RedactionMethod
    ) -> np.ndarray:
        """Apply redaction to a single bounding box region.
        
        Args:
            frame: Input frame
            bbox: Bounding box to redact
            method: Redaction method to apply
            
        Returns:
            Frame with redaction applied to the specified region
        """
        # Inflate bounding box for better coverage
        inflated_bbox = bbox.inflate(self.config.inflate_bbox_px)
        
        # Clamp coordinates to frame boundaries
        h, w = frame.shape[:2]
        x1 = max(0, min(inflated_bbox.x1, w - 1))
        y1 = max(0, min(inflated_bbox.y1, h - 1))
        x2 = max(x1 + 1, min(inflated_bbox.x2, w))
        y2 = max(y1 + 1, min(inflated_bbox.y2, h))
        
        # Skip if region is too small
        if x2 - x1 < 2 or y2 - y1 < 2:
            return frame
            
        # Extract region of interest
        roi = frame[y1:y2, x1:x2].copy()
        
        # Apply redaction method
        if method == RedactionMethod.GAUSSIAN:
            redacted_roi = self._apply_gaussian_blur(roi)
        elif method == RedactionMethod.PIXELATE:
            redacted_roi = self._apply_pixelation(roi)
        elif method == RedactionMethod.SOLID:
            redacted_roi = self._apply_solid_color(roi)
        else:
            raise ValueError(f"Unsupported redaction method: {method}")
            
        # Replace region in frame
        frame[y1:y2, x1:x2] = redacted_roi
        
        return frame
        
    def _apply_gaussian_blur(self, roi: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to region of interest.
        
        Args:
            roi: Region of interest to blur
            
        Returns:
            Blurred region
        """
        kernel_size = self.config.gaussian_kernel_size
        sigma = self.config.gaussian_sigma
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(
            roi, 
            (kernel_size, kernel_size), 
            sigma, 
            sigma
        )
        
        return blurred
        
    def _apply_pixelation(self, roi: np.ndarray) -> np.ndarray:
        """Apply pixelation to region of interest.
        
        Args:
            roi: Region of interest to pixelate
            
        Returns:
            Pixelated region
        """
        h, w = roi.shape[:2]
        block_size = self.config.pixelate_block_size
        
        # Skip if region is smaller than block size
        if h < block_size or w < block_size:
            return roi
            
        # Downsample by block size
        small_h = max(1, h // block_size)
        small_w = max(1, w // block_size)
        
        # Resize down then back up for pixelation effect
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return pixelated
        
    def _apply_solid_color(self, roi: np.ndarray) -> np.ndarray:
        """Apply solid color to region of interest.
        
        Args:
            roi: Region of interest to fill
            
        Returns:
            Solid color region
        """
        # Create solid color array with same shape as ROI
        color = self.config.solid_color
        solid = np.full_like(roi, color, dtype=roi.dtype)
        
        return solid
        
    def get_method_for_category(self, category: str) -> RedactionMethod:
        """Get the redaction method configured for a specific category.
        
        Args:
            category: Detection category name
            
        Returns:
            Redaction method for the category
        """
        method_name = self.config.category_methods.get(
            category, 
            self.config.default_method
        )
        return RedactionMethod(method_name)
        
    def set_method_for_category(self, category: str, method: RedactionMethod) -> None:
        """Set the redaction method for a specific category.
        
        Args:
            category: Detection category name
            method: Redaction method to use
        """
        self.config.category_methods[category] = method.value
        
    def get_supported_methods(self) -> List[str]:
        """Get list of supported redaction method names.
        
        Returns:
            List of method names
        """
        return [method.value for method in RedactionMethod]