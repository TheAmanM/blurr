"""Core data types for privacy redactor real-time processing."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class BBox:
    """Bounding box with confidence score."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def __post_init__(self):
        """Validate bounding box coordinates."""
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("Invalid bounding box: x1 must be < x2 and y1 must be < y2")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    @property
    def width(self) -> int:
        """Get bounding box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Get bounding box height."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Get bounding box area."""
        return self.width * self.height

    def iou(self, other: 'BBox') -> float:
        """Calculate Intersection over Union with another bounding box."""
        # Calculate intersection coordinates
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        # No intersection
        if x1 >= x2 or y1 >= y2:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0

    def inflate(self, pixels: int) -> 'BBox':
        """Create new bounding box inflated by specified pixels."""
        return BBox(
            x1=max(0, self.x1 - pixels),
            y1=max(0, self.y1 - pixels),
            x2=self.x2 + pixels,
            y2=self.y2 + pixels,
            confidence=self.confidence
        )


@dataclass
class Detection:
    """Text detection result with optional OCR text."""
    bbox: BBox
    text: Optional[str]
    timestamp: float

    def __post_init__(self):
        """Validate detection data."""
        if self.timestamp < 0:
            raise ValueError("Timestamp must be non-negative")


@dataclass
class Match:
    """Classification match for sensitive data."""
    category: str
    confidence: float
    masked_text: str
    bbox: BBox

    def __post_init__(self):
        """Validate match data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.category:
            raise ValueError("Category cannot be empty")


@dataclass
class Track:
    """Temporal track for consistent detection across frames."""
    id: str
    bbox: BBox
    matches: List[Match]
    age: int
    hits: int
    last_ocr_frame: int
    flow_points: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate track data."""
        if self.age < 0:
            raise ValueError("Age must be non-negative")
        if self.hits < 0:
            raise ValueError("Hits must be non-negative")
        if self.last_ocr_frame < -1:
            raise ValueError("Last OCR frame must be >= -1")
        if not self.id:
            raise ValueError("Track ID cannot be empty")

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate (hits / age)."""
        return self.hits / max(1, self.age)

    def update_bbox(self, new_bbox: BBox, smoothing_factor: float = 0.3) -> None:
        """Update bounding box with smoothing."""
        if smoothing_factor <= 0:
            self.bbox = new_bbox
        else:
            # Apply exponential moving average for smoothing
            self.bbox = BBox(
                x1=int(self.bbox.x1 * (1 - smoothing_factor) + new_bbox.x1 * smoothing_factor),
                y1=int(self.bbox.y1 * (1 - smoothing_factor) + new_bbox.y1 * smoothing_factor),
                x2=int(self.bbox.x2 * (1 - smoothing_factor) + new_bbox.x2 * smoothing_factor),
                y2=int(self.bbox.y2 * (1 - smoothing_factor) + new_bbox.y2 * smoothing_factor),
                confidence=max(self.bbox.confidence, new_bbox.confidence)
            )

    def add_match(self, match: Match) -> None:
        """Add a classification match to this track."""
        # Remove existing matches of the same category
        self.matches = [m for m in self.matches if m.category != match.category]
        self.matches.append(match)

    def get_best_match(self) -> Optional[Match]:
        """Get the match with highest confidence."""
        if not self.matches:
            return None
        return max(self.matches, key=lambda m: m.confidence)