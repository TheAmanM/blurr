"""Unit tests for core data types."""

import pytest
import numpy as np
from privacy_redactor_rt.types import BBox, Detection, Match, Track


class TestBBox:
    """Test BBox dataclass."""

    def test_valid_bbox_creation(self):
        """Test creating valid bounding box."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 50
        assert bbox.y2 == 60
        assert bbox.confidence == 0.8

    def test_invalid_coordinates(self):
        """Test invalid coordinate validation."""
        with pytest.raises(ValueError, match="Invalid bounding box"):
            BBox(x1=50, y1=20, x2=10, y2=60, confidence=0.8)
        
        with pytest.raises(ValueError, match="Invalid bounding box"):
            BBox(x1=10, y1=60, x2=50, y2=20, confidence=0.8)

    def test_invalid_confidence(self):
        """Test invalid confidence validation."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            BBox(x1=10, y1=20, x2=50, y2=60, confidence=1.5)
        
        with pytest.raises(ValueError, match="Confidence must be between"):
            BBox(x1=10, y1=20, x2=50, y2=60, confidence=-0.1)

    def test_properties(self):
        """Test bounding box properties."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        assert bbox.width == 40
        assert bbox.height == 40
        assert bbox.area == 1600

    def test_iou_calculation(self):
        """Test IoU calculation between bounding boxes."""
        bbox1 = BBox(x1=0, y1=0, x2=10, y2=10, confidence=0.8)
        bbox2 = BBox(x1=5, y1=5, x2=15, y2=15, confidence=0.9)
        
        # Expected IoU: intersection=25, union=175, iou=25/175≈0.143
        iou = bbox1.iou(bbox2)
        assert abs(iou - 0.142857) < 0.001

    def test_iou_no_overlap(self):
        """Test IoU with no overlap."""
        bbox1 = BBox(x1=0, y1=0, x2=10, y2=10, confidence=0.8)
        bbox2 = BBox(x1=20, y1=20, x2=30, y2=30, confidence=0.9)
        
        assert bbox1.iou(bbox2) == 0.0

    def test_iou_identical(self):
        """Test IoU with identical boxes."""
        bbox1 = BBox(x1=0, y1=0, x2=10, y2=10, confidence=0.8)
        bbox2 = BBox(x1=0, y1=0, x2=10, y2=10, confidence=0.9)
        
        assert bbox1.iou(bbox2) == 1.0

    def test_inflate(self):
        """Test bounding box inflation."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        inflated = bbox.inflate(5)
        
        assert inflated.x1 == 5
        assert inflated.y1 == 15
        assert inflated.x2 == 55
        assert inflated.y2 == 65
        assert inflated.confidence == 0.8

    def test_inflate_boundary(self):
        """Test inflation at boundaries."""
        bbox = BBox(x1=2, y1=3, x2=10, y2=15, confidence=0.8)
        inflated = bbox.inflate(5)
        
        # Should not go below 0
        assert inflated.x1 == 0
        assert inflated.y1 == 0
        assert inflated.x2 == 15
        assert inflated.y2 == 20


class TestDetection:
    """Test Detection dataclass."""

    def test_valid_detection(self):
        """Test creating valid detection."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        detection = Detection(bbox=bbox, text="test text", timestamp=123.45)
        
        assert detection.bbox == bbox
        assert detection.text == "test text"
        assert detection.timestamp == 123.45

    def test_detection_without_text(self):
        """Test detection without OCR text."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        detection = Detection(bbox=bbox, text=None, timestamp=123.45)
        
        assert detection.bbox == bbox
        assert detection.text is None
        assert detection.timestamp == 123.45

    def test_invalid_timestamp(self):
        """Test invalid timestamp validation."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        
        with pytest.raises(ValueError, match="Timestamp must be non-negative"):
            Detection(bbox=bbox, text="test", timestamp=-1.0)


class TestMatch:
    """Test Match dataclass."""

    def test_valid_match(self):
        """Test creating valid match."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        match = Match(
            category="phone",
            confidence=0.9,
            masked_text="555-***-****",
            bbox=bbox
        )
        
        assert match.category == "phone"
        assert match.confidence == 0.9
        assert match.masked_text == "555-***-****"
        assert match.bbox == bbox

    def test_invalid_confidence(self):
        """Test invalid confidence validation."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        
        with pytest.raises(ValueError, match="Confidence must be between"):
            Match(category="phone", confidence=1.5, masked_text="test", bbox=bbox)

    def test_empty_category(self):
        """Test empty category validation."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        
        with pytest.raises(ValueError, match="Category cannot be empty"):
            Match(category="", confidence=0.9, masked_text="test", bbox=bbox)


class TestTrack:
    """Test Track dataclass."""

    def test_valid_track(self):
        """Test creating valid track."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        track = Track(
            id="track_1",
            bbox=bbox,
            matches=[],
            age=5,
            hits=3,
            last_ocr_frame=10
        )
        
        assert track.id == "track_1"
        assert track.bbox == bbox
        assert track.matches == []
        assert track.age == 5
        assert track.hits == 3
        assert track.last_ocr_frame == 10
        assert track.flow_points is None

    def test_track_with_flow_points(self):
        """Test track with optical flow points."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        flow_points = np.array([[1, 2], [3, 4]], dtype=np.float32)
        
        track = Track(
            id="track_1",
            bbox=bbox,
            matches=[],
            age=5,
            hits=3,
            last_ocr_frame=10,
            flow_points=flow_points
        )
        
        assert np.array_equal(track.flow_points, flow_points)

    def test_invalid_values(self):
        """Test invalid value validation."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        
        with pytest.raises(ValueError, match="Age must be non-negative"):
            Track(id="track_1", bbox=bbox, matches=[], age=-1, hits=3, last_ocr_frame=10)
        
        with pytest.raises(ValueError, match="Hits must be non-negative"):
            Track(id="track_1", bbox=bbox, matches=[], age=5, hits=-1, last_ocr_frame=10)
        
        with pytest.raises(ValueError, match="Last OCR frame must be >= -1"):
            Track(id="track_1", bbox=bbox, matches=[], age=5, hits=3, last_ocr_frame=-2)
        
        with pytest.raises(ValueError, match="Track ID cannot be empty"):
            Track(id="", bbox=bbox, matches=[], age=5, hits=3, last_ocr_frame=10)

    def test_hit_rate(self):
        """Test hit rate calculation."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        track = Track(id="track_1", bbox=bbox, matches=[], age=10, hits=7, last_ocr_frame=5)
        
        assert track.hit_rate == 0.7

    def test_hit_rate_zero_age(self):
        """Test hit rate with zero age."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        track = Track(id="track_1", bbox=bbox, matches=[], age=0, hits=0, last_ocr_frame=-1)
        
        assert track.hit_rate == 0.0

    def test_update_bbox_no_smoothing(self):
        """Test bbox update without smoothing."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        new_bbox = BBox(x1=15, y1=25, x2=55, y2=65, confidence=0.9)
        
        track = Track(id="track_1", bbox=bbox, matches=[], age=5, hits=3, last_ocr_frame=10)
        track.update_bbox(new_bbox, smoothing_factor=0.0)
        
        assert track.bbox == new_bbox

    def test_update_bbox_with_smoothing(self):
        """Test bbox update with smoothing."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        new_bbox = BBox(x1=20, y1=30, x2=60, y2=70, confidence=0.9)
        
        track = Track(id="track_1", bbox=bbox, matches=[], age=5, hits=3, last_ocr_frame=10)
        track.update_bbox(new_bbox, smoothing_factor=0.5)
        
        # Expected: (10+20)/2=15, (20+30)/2=25, etc.
        assert track.bbox.x1 == 15
        assert track.bbox.y1 == 25
        assert track.bbox.x2 == 55
        assert track.bbox.y2 == 65
        assert track.bbox.confidence == 0.9  # Max confidence

    def test_add_match(self):
        """Test adding matches to track."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        track = Track(id="track_1", bbox=bbox, matches=[], age=5, hits=3, last_ocr_frame=10)
        
        match1 = Match(category="phone", confidence=0.9, masked_text="555-***-****", bbox=bbox)
        match2 = Match(category="email", confidence=0.8, masked_text="***@***.com", bbox=bbox)
        
        track.add_match(match1)
        assert len(track.matches) == 1
        assert track.matches[0] == match1
        
        track.add_match(match2)
        assert len(track.matches) == 2

    def test_add_match_replace_category(self):
        """Test replacing match of same category."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        track = Track(id="track_1", bbox=bbox, matches=[], age=5, hits=3, last_ocr_frame=10)
        
        match1 = Match(category="phone", confidence=0.9, masked_text="555-***-****", bbox=bbox)
        match2 = Match(category="phone", confidence=0.95, masked_text="666-***-****", bbox=bbox)
        
        track.add_match(match1)
        track.add_match(match2)
        
        assert len(track.matches) == 1
        assert track.matches[0] == match2

    def test_get_best_match(self):
        """Test getting best match by confidence."""
        bbox = BBox(x1=10, y1=20, x2=50, y2=60, confidence=0.8)
        track = Track(id="track_1", bbox=bbox, matches=[], age=5, hits=3, last_ocr_frame=10)
        
        # No matches
        assert track.get_best_match() is None
        
        match1 = Match(category="phone", confidence=0.8, masked_text="555-***-****", bbox=bbox)
        match2 = Match(category="email", confidence=0.9, masked_text="***@***.com", bbox=bbox)
        
        track.add_match(match1)
        track.add_match(match2)
        
        best = track.get_best_match()
        assert best == match2  # Higher confidence