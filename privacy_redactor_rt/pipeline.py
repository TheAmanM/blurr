"""Real-time processing pipeline orchestrator."""

import logging
import time
from typing import List, Optional, Dict, Any
import numpy as np

from .types import Detection, Track, BBox, Match
from .config import Config
from .text_detect import TextDetector
from .ocr import OCRWorker
from .classify import ClassificationEngine
from .track import OpticalFlowTracker
from .redact import RedactionEngine

logger = logging.getLogger(__name__)


class RealtimePipeline:
    """Orchestrates frame-by-frame processing with temporal consensus."""
    
    def __init__(self, config: Config):
        """Initialize processing pipeline with all components.
        
        Args:
            config: Complete configuration object
        """
        self.config = config
        self.frame_count = 0
        self.prev_frame: Optional[np.ndarray] = None
        
        # Initialize processing components
        self.text_detector = TextDetector(config.detection)
        self.ocr_worker = OCRWorker(config.ocr, config.realtime.max_queue)
        self.classifier = ClassificationEngine(config.classification)
        self.tracker = OpticalFlowTracker(config.tracking)
        self.redactor = RedactionEngine(config.redaction)
        
        # Temporal consensus tracking
        self.consensus_buffer: Dict[str, List[Match]] = {}  # track_id -> recent matches
        self.consensus_required = config.classification.require_temporal_consensus
        
        # Performance tracking
        self.stats = {
            'frames_processed': 0,
            'detections_run': 0,
            'ocr_requests': 0,
            'tracks_active': 0,
            'avg_processing_time': 0.0
        }
        
        logger.info("RealtimePipeline initialized with all components")
    
    def start(self) -> None:
        """Start the pipeline and all worker threads."""
        self.ocr_worker.start()
        logger.info("Pipeline started")
    
    def stop(self) -> None:
        """Stop the pipeline and cleanup resources."""
        self.ocr_worker.stop()
        logger.info("Pipeline stopped")
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Process single frame through the complete pipeline.
        
        Args:
            frame: Input frame as BGR numpy array
            frame_idx: Frame index for scheduling decisions
            
        Returns:
            Processed frame with redactions applied
        """
        start_time = time.time()
        self.frame_count = frame_idx
        
        # Validate input frame
        if frame is None or frame.size == 0:
            logger.warning(f"Invalid frame {frame_idx}: frame is None or empty")
            return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Ensure frame has proper dimensions
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            logger.warning(f"Invalid frame {frame_idx}: expected 3-channel BGR, got shape {frame.shape}")
            return frame
        
        try:
            # Step 1: Propagate existing tracks using optical flow
            if self.prev_frame is not None and self.prev_frame.shape == frame.shape:
                self.tracker.propagate_tracks(frame, self.prev_frame)
            else:
                # First frame or dimension change, just update prev_frame
                self.tracker.propagate_tracks(frame, None)
            
            # Step 2: Run text detection if scheduled
            detections = []
            if self.should_run_detection(frame_idx):
                detections = self._run_text_detection(frame)
                self.stats['detections_run'] += 1
            
            # Step 3: Associate detections with tracks
            if detections:
                detection_objects = [
                    Detection(bbox=bbox, text=None, timestamp=time.time())
                    for bbox in detections
                ]
                self.tracker.associate_detections(detection_objects)
            
            # Step 4: Schedule OCR for tracks that need it
            self._schedule_ocr_processing(frame)
            
            # Step 5: Classify OCR results and update track matches
            self._update_track_classifications()
            
            # Step 6: Apply temporal consensus filtering
            active_tracks = self._get_consensus_tracks()
            
            # Step 7: Apply redaction to final tracks
            redacted_frame = self.redactor.redact_regions(frame, active_tracks)
            
            # Step 8: Cleanup and update state
            self.tracker.cleanup_tracks()
            self._cleanup_consensus_buffer()
            
            # Safely copy frame for next iteration
            try:
                if frame.flags['C_CONTIGUOUS']:
                    self.prev_frame = frame.copy()
                else:
                    self.prev_frame = np.ascontiguousarray(frame)
            except Exception as e:
                logger.warning(f"Failed to copy frame: {e}")
                self.prev_frame = None
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(processing_time, len(active_tracks))
            
            return redacted_frame
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {e}")
            # Return original frame on error
            return frame
    
    def should_run_detection(self, frame_idx: int) -> bool:
        """Determine if text detection should run on this frame.
        
        Args:
            frame_idx: Current frame index
            
        Returns:
            True if detection should run
        """
        # Run detection every N frames based on detector_stride
        return frame_idx % self.config.realtime.detector_stride == 0
    
    def update_tracks(self, detections: List[Detection]) -> None:
        """Update tracking state with new detections.
        
        Args:
            detections: List of new detections to process
        """
        self.tracker.associate_detections(detections)
    
    def _run_text_detection(self, frame: np.ndarray) -> List[BBox]:
        """Run text detection on the current frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected text bounding boxes
        """
        try:
            bboxes = self.text_detector.detect(frame)
            logger.debug(f"Detected {len(bboxes)} text regions")
            return bboxes
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            return []
    
    def _schedule_ocr_processing(self, frame: np.ndarray) -> None:
        """Schedule OCR processing for tracks that need it.
        
        Args:
            frame: Current frame for ROI extraction
        """
        active_tracks = self.tracker.get_all_tracks()
        
        for track in active_tracks:
            # Check if track needs OCR
            if self._should_ocr_track(track):
                # Extract ROI from frame
                roi = self._extract_roi(frame, track.bbox)
                if roi is not None:
                    # Enqueue for OCR processing
                    success = self.ocr_worker.enqueue_roi(roi, track.id, track.bbox)
                    if success:
                        self.tracker.update_track_ocr_frame(track.id, self.frame_count)
                        self.stats['ocr_requests'] += 1
    
    def _should_ocr_track(self, track: Track) -> bool:
        """Determine if a track needs OCR processing.
        
        Args:
            track: Track to check
            
        Returns:
            True if OCR should be performed
        """
        # Use tracker's built-in logic
        return self.tracker.should_ocr_track(
            track.id, 
            self.frame_count, 
            self.config.realtime.ocr_refresh_stride
        )
    
    def _extract_roi(self, frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        """Extract region of interest from frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box coordinates
            
        Returns:
            ROI as numpy array or None if invalid
        """
        try:
            h, w = frame.shape[:2]
            
            # Clamp coordinates to frame bounds
            x1 = max(0, min(bbox.x1, w - 1))
            y1 = max(0, min(bbox.y1, h - 1))
            x2 = max(x1 + 1, min(bbox.x2, w))
            y2 = max(y1 + 1, min(bbox.y2, h))
            
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            
            # Validate ROI size
            if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
                return None
                
            return roi
            
        except Exception as e:
            logger.warning(f"Failed to extract ROI: {e}")
            return None
    
    def _update_track_classifications(self) -> None:
        """Update track classifications with OCR results."""
        active_tracks = self.tracker.get_all_tracks()
        
        for track in active_tracks:
            # Get OCR result for this track
            ocr_text = self.ocr_worker.get_result(track.id)
            if ocr_text:
                # Classify the text
                matches = self.classifier.classify_text(ocr_text, track.bbox)
                
                # Update track with new matches
                for match in matches:
                    track.add_match(match)
                
                # Add to consensus buffer
                if matches:
                    if track.id not in self.consensus_buffer:
                        self.consensus_buffer[track.id] = []
                    self.consensus_buffer[track.id].extend(matches)
                    
                    # Keep only recent matches (last N frames)
                    max_history = self.consensus_required * 2
                    if len(self.consensus_buffer[track.id]) > max_history:
                        self.consensus_buffer[track.id] = self.consensus_buffer[track.id][-max_history:]
    
    def _get_consensus_tracks(self) -> List[Track]:
        """Get tracks that meet temporal consensus requirements.
        
        Returns:
            List of tracks with confirmed detections
        """
        consensus_tracks = []
        active_tracks = self.tracker.get_active_tracks()
        
        for track in active_tracks:
            if self._has_temporal_consensus(track):
                consensus_tracks.append(track)
        
        return consensus_tracks
    
    def _has_temporal_consensus(self, track: Track) -> bool:
        """Check if track has sufficient temporal consensus.
        
        Args:
            track: Track to check
            
        Returns:
            True if track meets consensus requirements
        """
        if track.id not in self.consensus_buffer:
            return False
        
        recent_matches = self.consensus_buffer[track.id]
        if len(recent_matches) < self.consensus_required:
            return False
        
        # Group matches by category
        category_counts: Dict[str, int] = {}
        for match in recent_matches[-self.consensus_required:]:
            category_counts[match.category] = category_counts.get(match.category, 0) + 1
        
        # Check if any category has enough consecutive matches
        for category, count in category_counts.items():
            if count >= self.consensus_required:
                return True
        
        return False
    
    def _cleanup_consensus_buffer(self) -> None:
        """Clean up old entries from consensus buffer."""
        # Remove entries for tracks that no longer exist
        active_track_ids = set(track.id for track in self.tracker.get_all_tracks())
        
        expired_ids = []
        for track_id in self.consensus_buffer:
            if track_id not in active_track_ids:
                expired_ids.append(track_id)
        
        for track_id in expired_ids:
            del self.consensus_buffer[track_id]
    
    def _update_stats(self, processing_time: float, active_tracks: int) -> None:
        """Update performance statistics.
        
        Args:
            processing_time: Time taken to process current frame
            active_tracks: Number of active tracks
        """
        self.stats['frames_processed'] += 1
        self.stats['tracks_active'] = active_tracks
        
        # Update average processing time with exponential moving average
        alpha = 0.1
        if self.stats['avg_processing_time'] == 0:
            self.stats['avg_processing_time'] = processing_time
        else:
            self.stats['avg_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats['avg_processing_time']
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        ocr_stats = self.ocr_worker.get_stats()
        
        return {
            **self.stats,
            'ocr_queue_size': ocr_stats['queue_size'],
            'ocr_cache_size': ocr_stats['cache_size'],
            'ocr_processed': ocr_stats['processed_count'],
            'ocr_queue_full': ocr_stats['queue_full_count'],
            'consensus_buffer_size': len(self.consensus_buffer),
            'total_tracks': len(self.tracker.tracks)
        }
    
    def reset(self) -> None:
        """Reset pipeline state."""
        self.frame_count = 0
        self.prev_frame = None
        self.consensus_buffer.clear()
        self.tracker.reset()
        
        # Reset statistics
        self.stats = {
            'frames_processed': 0,
            'detections_run': 0,
            'ocr_requests': 0,
            'tracks_active': 0,
            'avg_processing_time': 0.0
        }
        
        logger.info("Pipeline state reset")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()