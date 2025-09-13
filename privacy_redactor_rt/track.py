"""Optical flow tracker with IoU association for temporal consistency."""

import uuid
from typing import Dict, List, Optional, Tuple
import numpy as np

# Check if cv2 is available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

from .types import BBox, Detection, Track
from .config import TrackingConfig


class OpticalFlowTracker:
    """Tracks bounding boxes across frames using optical flow and IoU association."""
    
    def __init__(self, config: TrackingConfig):
        """Initialize the optical flow tracker.
        
        Args:
            config: Tracking configuration parameters
        """
        self.config = config
        self.tracks: Dict[str, Track] = {}
        self.frame_count = 0
        self.prev_gray: Optional[np.ndarray] = None
        
        # Lucas-Kanade optical flow parameters
        if CV2_AVAILABLE:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        else:
            criteria = (1 | 2, 10, 0.03)  # Fallback for testing
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=criteria
        )
        
        # Good features to track parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=config.flow_quality_level,
            minDistance=config.flow_min_distance,
            blockSize=config.flow_block_size
        )
    
    def propagate_tracks(self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> None:
        """Propagate existing tracks using optical flow.
        
        Args:
            frame: Current frame (BGR format)
            prev_frame: Previous frame (BGR format), if None uses stored prev_gray
        """
        if len(self.tracks) == 0:
            self._update_prev_frame(frame)
            return
        
        # Convert to grayscale
        if CV2_AVAILABLE:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            # Fallback for testing - assume frame is already grayscale or convert manually
            if len(frame.shape) == 3:
                gray = np.mean(frame, axis=2).astype(np.uint8)
            else:
                gray = frame
        
        if prev_frame is not None:
            if CV2_AVAILABLE:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            else:
                # Fallback for testing
                if len(prev_frame.shape) == 3:
                    prev_gray = np.mean(prev_frame, axis=2).astype(np.uint8)
                else:
                    prev_gray = prev_frame
        else:
            prev_gray = self.prev_gray
        
        if prev_gray is None:
            self._update_prev_frame(frame)
            return
        
        # Propagate each track
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.flow_points is None or len(track.flow_points) == 0:
                # Initialize flow points for this track
                self._initialize_flow_points(track, prev_gray)
                continue
            
            # Calculate optical flow
            if CV2_AVAILABLE:
                new_points, status, error = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, track.flow_points, None, **self.lk_params
                )
            else:
                # Fallback for testing - simulate optical flow
                new_points = track.flow_points + np.random.normal(0, 1, track.flow_points.shape)
                status = np.ones((len(track.flow_points), 1), dtype=np.uint8)
                error = np.ones((len(track.flow_points), 1), dtype=np.float32)
            
            # Filter good points
            good_mask = status.ravel() == 1
            good_new = new_points[good_mask]
            good_old = track.flow_points[good_mask]
            
            if len(good_new) < 3:  # Need at least 3 points for reliable tracking
                # Mark track for aging
                track.age += 1
                if track.age > self.config.max_age:
                    tracks_to_remove.append(track_id)
                continue
            
            # Calculate average displacement
            displacement = np.mean(good_new - good_old, axis=0)
            
            # Check if displacement is reasonable
            displacement_magnitude = np.linalg.norm(displacement)
            if displacement_magnitude > self.config.max_flow_error:
                # Large displacement, likely tracking failure
                track.age += 1
                if track.age > self.config.max_age:
                    tracks_to_remove.append(track_id)
                continue
            
            # Update bounding box based on optical flow
            new_bbox = BBox(
                x1=int(track.bbox.x1 + displacement[0]),
                y1=int(track.bbox.y1 + displacement[1]),
                x2=int(track.bbox.x2 + displacement[0]),
                y2=int(track.bbox.y2 + displacement[1]),
                confidence=track.bbox.confidence
            )
            
            # Ensure bbox is within frame bounds
            h, w = gray.shape
            new_bbox = BBox(
                x1=max(0, min(w - 1, new_bbox.x1)),
                y1=max(0, min(h - 1, new_bbox.y1)),
                x2=max(1, min(w, new_bbox.x2)),
                y2=max(1, min(h, new_bbox.y2)),
                confidence=new_bbox.confidence
            )
            
            # Apply smoothing
            track.update_bbox(new_bbox, self.config.smoothing_factor)
            
            # Update flow points
            track.flow_points = good_new
            track.age += 1
            
            # Remove old tracks
            if track.age > self.config.max_age:
                tracks_to_remove.append(track_id)
        
        # Remove expired tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        self._update_prev_frame(frame)
    
    def associate_detections(self, detections: List[Detection]) -> None:
        """Associate new detections with existing tracks using IoU.
        
        Args:
            detections: List of new detections to associate
        """
        self.frame_count += 1
        
        if not detections:
            return
        
        # Convert detections to bboxes for easier processing
        detection_bboxes = [det.bbox for det in detections]
        
        # Calculate IoU matrix between tracks and detections
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detection_bboxes)))
        
        for i, track_id in enumerate(track_ids):
            track_bbox = self.tracks[track_id].bbox
            for j, det_bbox in enumerate(detection_bboxes):
                iou_matrix[i, j] = track_bbox.iou(det_bbox)
        
        # Hungarian algorithm would be ideal, but for simplicity use greedy matching
        matched_tracks = set()
        matched_detections = set()
        
        # Sort by IoU score (highest first)
        matches = []
        for i in range(len(track_ids)):
            for j in range(len(detection_bboxes)):
                if iou_matrix[i, j] >= self.config.iou_threshold:
                    matches.append((i, j, iou_matrix[i, j]))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Assign matches greedily
        for track_idx, det_idx, iou_score in matches:
            if track_idx not in matched_tracks and det_idx not in matched_detections:
                track_id = track_ids[track_idx]
                track = self.tracks[track_id]
                detection = detections[det_idx]
                
                # Update track with new detection
                track.update_bbox(detection.bbox, self.config.smoothing_factor)
                track.hits += 1
                track.age = 0  # Reset age on successful match
                
                # Update text if available
                if detection.text is not None:
                    # Store the detection for later classification
                    pass
                
                matched_tracks.add(track_idx)
                matched_detections.add(det_idx)
        
        # Create new tracks for unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                track_id = str(uuid.uuid4())
                new_track = Track(
                    id=track_id,
                    bbox=detection.bbox,
                    matches=[],
                    age=0,
                    hits=1,
                    last_ocr_frame=-1,
                    flow_points=None
                )
                self.tracks[track_id] = new_track
    
    def get_active_tracks(self) -> List[Track]:
        """Get tracks that meet minimum hit requirements.
        
        Returns:
            List of active tracks that should be displayed/processed
        """
        active_tracks = []
        for track in self.tracks.values():
            # Track is active if it has enough hits or good hit rate
            if (track.hits >= self.config.min_hits or 
                (track.age > 0 and track.hit_rate >= 0.5)):
                active_tracks.append(track)
        
        return active_tracks
    
    def get_all_tracks(self) -> List[Track]:
        """Get all current tracks regardless of status.
        
        Returns:
            List of all tracks
        """
        return list(self.tracks.values())
    
    def cleanup_tracks(self) -> None:
        """Remove expired and low-quality tracks."""
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            # Remove tracks that are too old
            if track.age > self.config.max_age:
                tracks_to_remove.append(track_id)
            # Remove tracks with very low hit rate
            elif track.age > 10 and track.hit_rate < 0.1:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self.frame_count = 0
        self.prev_gray = None
    
    def _initialize_flow_points(self, track: Track, gray_frame: np.ndarray) -> None:
        """Initialize optical flow points for a track.
        
        Args:
            track: Track to initialize points for
            gray_frame: Grayscale frame to extract features from
        """
        # Extract region of interest
        bbox = track.bbox
        roi = gray_frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        
        if roi.size == 0:
            return
        
        # Find good features to track in the ROI
        if CV2_AVAILABLE:
            corners = cv2.goodFeaturesToTrack(roi, **self.feature_params)
        else:
            # Fallback for testing - create some dummy corners
            corners = np.array([[[5, 5]], [[roi.shape[1]-5, 5]], 
                              [[5, roi.shape[0]-5]], [[roi.shape[1]-5, roi.shape[0]-5]]], 
                              dtype=np.float32)
        
        if corners is not None and len(corners) > 0:
            # Convert coordinates back to full frame
            corners[:, :, 0] += bbox.x1
            corners[:, :, 1] += bbox.y1
            track.flow_points = corners.reshape(-1, 2).astype(np.float32)
        else:
            # Fallback: use bbox corners
            track.flow_points = np.array([
                [bbox.x1, bbox.y1],
                [bbox.x2, bbox.y1],
                [bbox.x1, bbox.y2],
                [bbox.x2, bbox.y2]
            ], dtype=np.float32)
    
    def _update_prev_frame(self, frame: np.ndarray) -> None:
        """Update the previous frame for optical flow calculation.
        
        Args:
            frame: Current frame (BGR format)
        """
        if CV2_AVAILABLE:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            # Fallback for testing
            if len(frame.shape) == 3:
                self.prev_gray = np.mean(frame, axis=2).astype(np.uint8)
            else:
                self.prev_gray = frame
    
    def get_track_by_id(self, track_id: str) -> Optional[Track]:
        """Get track by ID.
        
        Args:
            track_id: Track identifier
            
        Returns:
            Track if found, None otherwise
        """
        return self.tracks.get(track_id)
    
    def update_track_ocr_frame(self, track_id: str, frame_number: int) -> None:
        """Update the last OCR frame for a track.
        
        Args:
            track_id: Track identifier
            frame_number: Frame number when OCR was performed
        """
        if track_id in self.tracks:
            self.tracks[track_id].last_ocr_frame = frame_number
    
    def should_ocr_track(self, track_id: str, current_frame: int, ocr_refresh_stride: int) -> bool:
        """Determine if a track needs OCR processing.
        
        Args:
            track_id: Track identifier
            current_frame: Current frame number
            ocr_refresh_stride: Frames between OCR refresh
            
        Returns:
            True if track should be OCR'd
        """
        track = self.tracks.get(track_id)
        if track is None:
            return False
        
        # OCR if never done before
        if track.last_ocr_frame == -1:
            return True
        
        # OCR if enough frames have passed
        if current_frame - track.last_ocr_frame >= ocr_refresh_stride:
            return True
        
        return False