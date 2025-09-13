"""Basic unit tests for optical flow tracker without OpenCV dependencies."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from privacy_redactor_rt.types import BBox, Detection, Track
from privacy_redactor_rt.config import TrackingConfig


class TestOpticalFlowTrackerBasic:
    """Test cases for OpticalFlowTracker class without OpenCV."""
    
    @pytest.fixture
    def config(self):
        """Create test tracking configuration."""
        return TrackingConfig(
            iou_threshold=0.5,
            max_age=30,
            min_hits=3,
            smoothing_factor=0.3,
            max_flow_error=50.0,
            flow_quality_level=0.01,
            flow_min_distance=10,
            flow_block_size=3
        )
    
    @pytest.fixture
    def sample_detection(self):
        """Create a sample detection."""
        bbox = BBox(x1=100, y1=100, x2=200, y2=200, confidence=0.8)
        return Detection(bbox=bbox, text="test text", timestamp=1.0)
    
    def test_tracker_init(self, config):
        """Test tracker initialization without OpenCV."""
        # Mock cv2 import
        with patch.dict('sys.modules', {'cv2': Mock()}):
            from privacy_redactor_rt.track import OpticalFlowTracker
            
            tracker = OpticalFlowTracker(config)
            
            assert tracker.config == config
            assert len(tracker.tracks) == 0
            assert tracker.frame_count == 0
            assert tracker.prev_gray is None
    
    def test_associate_detections_new_track(self, config, sample_detection):
        """Test creating new track from detection."""
        with patch.dict('sys.modules', {'cv2': Mock()}):
            from privacy_redactor_rt.track import OpticalFlowTracker
            
            tracker = OpticalFlowTracker(config)
            detections = [sample_detection]
            
            tracker.associate_detections(detections)
            
            assert len(tracker.tracks) == 1
            assert tracker.frame_count == 1
            
            track = list(tracker.tracks.values())[0]
            assert track.bbox.x1 == sample_detection.bbox.x1
            assert track.bbox.y1 == sample_detection.bbox.y1
            assert track.bbox.x2 == sample_detection.bbox.x2
            assert track.bbox.y2 == sample_detection.bbox.y2
            assert track.hits == 1
            assert track.age == 0
            assert track.last_ocr_frame == -1
    
    def test_associate_detections_match_existing(self, config, sample_detection):
        """Test matching detection with existing track."""
        with patch.dict('sys.modules', {'cv2': Mock()}):
            from privacy_redactor_rt.track import OpticalFlowTracker
            
            tracker = OpticalFlowTracker(config)
            
            # Create initial track
            detections = [sample_detection]
            tracker.associate_detections(detections)
            
            # Create slightly moved detection
            moved_bbox = BBox(x1=105, y1=105, x2=205, y2=205, confidence=0.9)
            moved_detection = Detection(bbox=moved_bbox, text="moved text", timestamp=2.0)
            
            tracker.associate_detections([moved_detection])
            
            # Should still have only one track
            assert len(tracker.tracks) == 1
            
            track = list(tracker.tracks.values())[0]
            assert track.hits == 2
            assert track.age == 0  # Reset on match
            
            # Bbox should be smoothed between original and new position
            assert track.bbox.x1 != sample_detection.bbox.x1  # Should be smoothed
            assert track.bbox.x1 != moved_detection.bbox.x1   # Should be smoothed
    
    def test_associate_detections_no_match(self, config, sample_detection):
        """Test detection that doesn't match existing track."""
        with patch.dict('sys.modules', {'cv2': Mock()}):
            from privacy_redactor_rt.track import OpticalFlowTracker
            
            tracker = OpticalFlowTracker(config)
            
            # Create initial track
            detections = [sample_detection]
            tracker.associate_detections(detections)
            
            # Create detection far away (no IoU overlap)
            far_bbox = BBox(x1=400, y1=400, x2=500, y2=500, confidence=0.8)
            far_detection = Detection(bbox=far_bbox, text="far text", timestamp=2.0)
            
            tracker.associate_detections([far_detection])
            
            # Should have two tracks now
            assert len(tracker.tracks) == 2
            
            # Check that both tracks exist
            bboxes = [track.bbox for track in tracker.tracks.values()]
            assert any(bbox.x1 == 100 for bbox in bboxes)  # Original
            assert any(bbox.x1 == 400 for bbox in bboxes)  # New
    
    def test_get_active_tracks(self, config):
        """Test filtering active tracks."""
        with patch.dict('sys.modules', {'cv2': Mock()}):
            from privacy_redactor_rt.track import OpticalFlowTracker
            
            tracker = OpticalFlowTracker(config)
            
            # Create track with enough hits
            bbox1 = BBox(x1=100, y1=100, x2=200, y2=200, confidence=0.8)
            track1 = Track(
                id="track1",
                bbox=bbox1,
                matches=[],
                age=5,
                hits=config.min_hits,
                last_ocr_frame=-1
            )
            
            # Create track with insufficient hits but good hit rate
            bbox2 = BBox(x1=300, y1=300, x2=400, y2=400, confidence=0.8)
            track2 = Track(
                id="track2",
                bbox=bbox2,
                matches=[],
                age=4,
                hits=2,  # Less than min_hits but hit_rate = 0.5
                last_ocr_frame=-1
            )
            
            # Create track with insufficient hits and poor hit rate
            bbox3 = BBox(x1=500, y1=500, x2=600, y2=600, confidence=0.8)
            track3 = Track(
                id="track3",
                bbox=bbox3,
                matches=[],
                age=10,
                hits=1,  # hit_rate = 0.1
                last_ocr_frame=-1
            )
            
            tracker.tracks = {"track1": track1, "track2": track2, "track3": track3}
            
            active_tracks = tracker.get_active_tracks()
            
            # Should return track1 (enough hits) and track2 (good hit rate)
            assert len(active_tracks) == 2
            active_ids = {track.id for track in active_tracks}
            assert "track1" in active_ids
            assert "track2" in active_ids
            assert "track3" not in active_ids
    
    def test_cleanup_tracks(self, config):
        """Test track cleanup logic."""
        with patch.dict('sys.modules', {'cv2': Mock()}):
            from privacy_redactor_rt.track import OpticalFlowTracker
            
            tracker = OpticalFlowTracker(config)
            
            # Create old track
            bbox1 = BBox(x1=100, y1=100, x2=200, y2=200, confidence=0.8)
            old_track = Track(
                id="old_track",
                bbox=bbox1,
                matches=[],
                age=config.max_age + 1,  # Too old
                hits=5,
                last_ocr_frame=-1
            )
            
            # Create track with poor hit rate
            bbox2 = BBox(x1=300, y1=300, x2=400, y2=400, confidence=0.8)
            poor_track = Track(
                id="poor_track",
                bbox=bbox2,
                matches=[],
                age=15,  # Old enough to check hit rate
                hits=1,   # Very poor hit rate
                last_ocr_frame=-1
            )
            
            # Create good track
            bbox3 = BBox(x1=500, y1=500, x2=600, y2=600, confidence=0.8)
            good_track = Track(
                id="good_track",
                bbox=bbox3,
                matches=[],
                age=10,
                hits=8,  # Good hit rate
                last_ocr_frame=-1
            )
            
            tracker.tracks = {
                "old_track": old_track,
                "poor_track": poor_track,
                "good_track": good_track
            }
            
            tracker.cleanup_tracks()
            
            # Should only keep the good track
            assert len(tracker.tracks) == 1
            assert "good_track" in tracker.tracks
    
    def test_reset(self, config, sample_detection):
        """Test tracker reset functionality."""
        with patch.dict('sys.modules', {'cv2': Mock()}):
            from privacy_redactor_rt.track import OpticalFlowTracker
            
            tracker = OpticalFlowTracker(config)
            
            # Add some state
            tracker.associate_detections([sample_detection])
            tracker.frame_count = 10
            tracker.prev_gray = np.zeros((480, 640), dtype=np.uint8)
            
            assert len(tracker.tracks) > 0
            assert tracker.frame_count > 0
            assert tracker.prev_gray is not None
            
            tracker.reset()
            
            assert len(tracker.tracks) == 0
            assert tracker.frame_count == 0
            assert tracker.prev_gray is None
    
    def test_get_track_by_id(self, config):
        """Test getting track by ID."""
        with patch.dict('sys.modules', {'cv2': Mock()}):
            from privacy_redactor_rt.track import OpticalFlowTracker
            
            tracker = OpticalFlowTracker(config)
            
            bbox = BBox(x1=100, y1=100, x2=200, y2=200, confidence=0.8)
            track = Track(
                id="test_track",
                bbox=bbox,
                matches=[],
                age=0,
                hits=1,
                last_ocr_frame=-1
            )
            tracker.tracks["test_track"] = track
            
            # Test existing track
            found_track = tracker.get_track_by_id("test_track")
            assert found_track is not None
            assert found_track.id == "test_track"
            
            # Test non-existing track
            not_found = tracker.get_track_by_id("nonexistent")
            assert not_found is None
    
    def test_update_track_ocr_frame(self, config):
        """Test updating track OCR frame."""
        with patch.dict('sys.modules', {'cv2': Mock()}):
            from privacy_redactor_rt.track import OpticalFlowTracker
            
            tracker = OpticalFlowTracker(config)
            
            bbox = BBox(x1=100, y1=100, x2=200, y2=200, confidence=0.8)
            track = Track(
                id="test_track",
                bbox=bbox,
                matches=[],
                age=0,
                hits=1,
                last_ocr_frame=-1
            )
            tracker.tracks["test_track"] = track
            
            tracker.update_track_ocr_frame("test_track", 42)
            
            assert tracker.tracks["test_track"].last_ocr_frame == 42
            
            # Test non-existing track (should not crash)
            tracker.update_track_ocr_frame("nonexistent", 100)
    
    def test_should_ocr_track(self, config):
        """Test OCR scheduling logic."""
        with patch.dict('sys.modules', {'cv2': Mock()}):
            from privacy_redactor_rt.track import OpticalFlowTracker
            
            tracker = OpticalFlowTracker(config)
            
            bbox = BBox(x1=100, y1=100, x2=200, y2=200, confidence=0.8)
            track = Track(
                id="test_track",
                bbox=bbox,
                matches=[],
                age=0,
                hits=1,
                last_ocr_frame=-1  # Never OCR'd
            )
            tracker.tracks["test_track"] = track
            
            # Should OCR if never done before
            assert tracker.should_ocr_track("test_track", 10, 5) is True
            
            # Update OCR frame
            tracker.update_track_ocr_frame("test_track", 10)
            
            # Should not OCR if recent
            assert tracker.should_ocr_track("test_track", 12, 5) is False
            
            # Should OCR if enough frames passed
            assert tracker.should_ocr_track("test_track", 16, 5) is True
            
            # Test non-existing track
            assert tracker.should_ocr_track("nonexistent", 10, 5) is False


class TestTrackingIntegrationBasic:
    """Basic integration tests for tracking functionality."""
    
    def test_full_tracking_sequence(self):
        """Test complete tracking sequence with multiple frames."""
        with patch.dict('sys.modules', {'cv2': Mock()}):
            from privacy_redactor_rt.track import OpticalFlowTracker
            
            config = TrackingConfig()
            tracker = OpticalFlowTracker(config)
            
            # Frame 1: Initial detection
            bbox1 = BBox(x1=100, y1=100, x2=200, y2=200, confidence=0.8)
            detection1 = Detection(bbox=bbox1, text="text1", timestamp=1.0)
            tracker.associate_detections([detection1])
            
            assert len(tracker.tracks) == 1
            track_id = list(tracker.tracks.keys())[0]
            
            # Frame 2: Slightly moved detection (should match)
            bbox2 = BBox(x1=105, y1=105, x2=205, y2=205, confidence=0.9)
            detection2 = Detection(bbox=bbox2, text="text2", timestamp=2.0)
            tracker.associate_detections([detection2])
            
            assert len(tracker.tracks) == 1  # Same track
            assert tracker.tracks[track_id].hits == 2
            
            # Frame 4: Detection returns (should match again)
            bbox4 = BBox(x1=110, y1=110, x2=210, y2=210, confidence=0.85)
            detection4 = Detection(bbox=bbox4, text="text4", timestamp=4.0)
            tracker.associate_detections([detection4])
            
            assert len(tracker.tracks) == 1  # Same track
            assert tracker.tracks[track_id].hits == 3
            assert tracker.tracks[track_id].age == 0  # Reset on match
            
            # Check if track is active
            active_tracks = tracker.get_active_tracks()
            assert len(active_tracks) == 1
            assert active_tracks[0].id == track_id