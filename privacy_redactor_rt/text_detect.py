"""Text detection wrapper with lazy initialization using PaddleOCR PP-OCRv4."""

import logging
from typing import List, Optional, Tuple
import numpy as np

from .config import DetectionConfig
from .types import BBox

logger = logging.getLogger(__name__)


class TextDetector:
    """Text detection wrapper using PaddleOCR PP-OCRv4 with lazy initialization."""
    
    def __init__(self, config: DetectionConfig):
        """Initialize TextDetector with configuration.
        
        Args:
            config: Detection configuration containing thresholds and parameters
        """
        self.config = config
        self._detector = None
        self._initialized = False
        
        logger.info(
            f"TextDetector initialized with config: "
            f"confidence={config.min_text_confidence}, "
            f"bbox_inflate={config.bbox_inflate_px}px, "
            f"min_size={config.min_box_size}, "
            f"max_size={config.max_box_size}"
        )
    
    def lazy_init(self) -> None:
        """Initialize PaddleOCR detector on first use to avoid startup delays."""
        if self._initialized:
            return
            
        try:
            # Import PaddleOCR only when needed
            from paddleocr import PaddleOCR
            
            logger.info("Initializing PaddleOCR PP-OCRv4 text detector...")
            
            # Initialize with detection only (no OCR)
            self._detector = PaddleOCR(
                lang='en',
                text_detection_model_dir=None,  # Use default detection model
                text_recognition_model_dir=None,  # Use default recognition model
                text_det_thresh=self.config.det_db_thresh,
                text_det_box_thresh=self.config.det_db_box_thresh,
                use_textline_orientation=False,  # Disable angle classification for speed
                ocr_version='PP-OCRv4'
            )
            
            self._initialized = True
            logger.info("PaddleOCR text detector initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import PaddleOCR: {e}")
            raise RuntimeError(
                "PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise RuntimeError(f"PaddleOCR initialization failed: {e}") from e
    
    def detect(self, frame: np.ndarray) -> List[BBox]:
        """Detect text regions in frame and return bounding boxes.
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            List of BBox objects with detected text regions
            
        Raises:
            RuntimeError: If detection fails
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided to text detector")
            return []
            
        # Lazy initialization on first detection
        self.lazy_init()
        
        try:
            # Run detection using the new predict API
            results = self._detector.predict(frame)
            
            if not results or len(results) == 0:
                return []
            
            # Extract bounding boxes from results
            bboxes = []
            
            # Results is a list, get the first result
            result = results[0]
            dt_polys = result.get('dt_polys', [])
            
            for detection in dt_polys:
                if detection is None:
                    continue
                    
                # PaddleOCR returns quadrilateral points as numpy array
                # Convert to list of [x, y] pairs
                if hasattr(detection, 'tolist'):
                    quad_points = detection.tolist()
                else:
                    quad_points = detection
                
                if len(quad_points) != 4:
                    continue
                
                # Convert quadrilateral to axis-aligned bounding box
                bbox = self._quad_to_bbox(quad_points)
                if bbox is None:
                    continue
                
                # Apply size filtering
                if not self._is_valid_size(bbox):
                    continue
                
                bboxes.append(bbox)
            
            logger.debug(f"Detected {len(bboxes)} text regions")
            return bboxes
            
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            raise RuntimeError(f"Text detection error: {e}") from e
    
    def _quad_to_bbox(self, quad_points: List[List[float]]) -> Optional[BBox]:
        """Convert quadrilateral points to axis-aligned bounding box.
        
        Args:
            quad_points: List of 4 [x, y] coordinate pairs
            
        Returns:
            BBox object or None if invalid
        """
        try:
            # Validate input
            if not quad_points or len(quad_points) != 4:
                return None
            
            # Extract x and y coordinates
            x_coords = [point[0] for point in quad_points]
            y_coords = [point[1] for point in quad_points]
            
            # Calculate axis-aligned bounding box
            x1 = int(min(x_coords))
            y1 = int(min(y_coords))
            x2 = int(max(x_coords))
            y2 = int(max(y_coords))
            
            # Validate coordinates (must have positive area)
            if x1 >= x2 or y1 >= y2:
                return None
            
            # Use default confidence since PaddleOCR detection doesn't provide it
            # in the current format (confidence is available in full OCR mode)
            confidence = self.config.min_text_confidence
            
            return BBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)
            
        except (IndexError, ValueError, TypeError) as e:
            logger.warning(f"Invalid quadrilateral points: {quad_points}, error: {e}")
            return None
    
    def _is_valid_size(self, bbox: BBox) -> bool:
        """Check if bounding box meets size requirements.
        
        Args:
            bbox: Bounding box to validate
            
        Returns:
            True if size is valid, False otherwise
        """
        width = bbox.width
        height = bbox.height
        
        # Check minimum size
        min_width, min_height = self.config.min_box_size
        if width < min_width or height < min_height:
            return False
        
        # Check maximum size
        max_width, max_height = self.config.max_box_size
        if width > max_width or height > max_height:
            return False
        
        return True
    
    def get_detector_info(self) -> dict:
        """Get information about the detector state.
        
        Returns:
            Dictionary with detector information
        """
        return {
            'initialized': self._initialized,
            'use_gpu': self.config.use_gpu,
            'min_confidence': self.config.min_text_confidence,
            'min_box_size': self.config.min_box_size,
            'max_box_size': self.config.max_box_size,
            'bbox_inflate_px': self.config.bbox_inflate_px,
            'det_db_thresh': self.config.det_db_thresh,
            'det_db_box_thresh': self.config.det_db_box_thresh
        }
    
    def __del__(self):
        """Cleanup detector resources."""
        if self._detector is not None:
            try:
                # PaddleOCR doesn't have explicit cleanup, but we can clear the reference
                self._detector = None
                logger.debug("TextDetector resources cleaned up")
            except Exception as e:
                logger.warning(f"Error during TextDetector cleanup: {e}")