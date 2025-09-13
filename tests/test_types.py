"""Test core data types."""

import numpy as np
from privacy_redactor_rt.types import BBox, Detection, Match, Track


def test_bbox_creation():
    """Test BBox dataclass creation."""
    bbox = BBox(x1=10, y1=20, x2=100, y2=200, confidence=0.95)
    assert bbox.x1 == 10
    assert bbox.y1 == 20
    assert bbox.x2 == 100
    assert bbox.y2 == 200
    assert bbox.confidence == 0.95


def test_detection_creation():
    """Test Detection dataclass creation."""
    bbox = BBox(x1=10, y1=20, x2=100, y2=200, confidence=0.95)
    detection = Detection(bbox=bbox, text="sample text", timestamp=123.456)
    assert detection.bbox == bbox
    assert detection.text == "sample text"
    assert detection.timestamp == 123.456


def test_match_creation():
    """Test Match dataclass creation."""
    bbox = BBox(x1=10, y1=20, x2=100, y2=200, confidence=0.95)
    match = Match(
        category="phone",
        confidence=0.85,
        masked_text="555-***-1234",
        bbox=bbox
    )
    assert match.category == "phone"
    assert match.confidence == 0.85
    assert match.masked_text == "555-***-1234"


def test_track_creation():
    """Test Track dataclass creation."""
    bbox = BBox(x1=10, y1=20, x2=100, y2=200, confidence=0.95)
    match = Match(
        category="phone",
        confidence=0.85,
        masked_text="555-***-1234",
        bbox=bbox
    )
    flow_points = np.array([[10, 20], [30, 40]])
    
    track = Track(
        id="track_001",
        bbox=bbox,
        matches=[match],
        age=5,
        hits=3,
        last_ocr_frame=10,
        flow_points=flow_points
    )
    assert track.id == "track_001"
    assert len(track.matches) == 1
    assert track.age == 5
    assert np.array_equal(track.flow_points, flow_points)