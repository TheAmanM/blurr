"""Configuration management using Pydantic models."""

from typing import List, Tuple, Dict, Any
from pydantic import BaseModel, Field
import yaml
from pathlib import Path


class IOConfig(BaseModel):
    """Input/Output configuration."""
    target_width: int = 1280
    target_height: int = 720
    target_fps: int = 30
    letterbox: bool = True


class RealtimeConfig(BaseModel):
    """Real-time processing configuration."""
    detector_stride: int = 3
    ocr_refresh_stride: int = 10
    max_parallel_ocr: int = 1
    max_queue: int = 2


class DetectionConfig(BaseModel):
    """Text detection configuration."""
    min_text_confidence: float = 0.6
    bbox_inflate_px: int = 6
    min_box_size: Tuple[int, int] = (10, 10)


class OCRConfig(BaseModel):
    """OCR processing configuration."""
    confidence_threshold: float = 0.7
    max_queue_size: int = 5
    timeout_seconds: float = 2.0


class ClassificationConfig(BaseModel):
    """Classification engine configuration."""
    require_temporal_consensus: int = 2
    categories: List[str] = ["phone", "credit_card", "email", "address", "api_key"]
    entropy_threshold_bits_per_char: float = 3.5


class TrackingConfig(BaseModel):
    """Optical flow tracking configuration."""
    iou_threshold: float = 0.5
    max_age: int = 30
    min_hits: int = 3
    smoothing_window: int = 5


class RedactionConfig(BaseModel):
    """Redaction engine configuration."""
    default_method: str = "gaussian"
    gaussian_kernel_size: int = 15
    pixelate_block_size: int = 8
    solid_color: Tuple[int, int, int] = (0, 0, 0)
    per_category_methods: Dict[str, str] = {}


class LoggingConfig(BaseModel):
    """Logging configuration."""
    enable_audit_log: bool = True
    log_text_previews: bool = True
    mask_characters: bool = True
    log_file_path: str = "detections.jsonl"


class Config(BaseModel):
    """Main configuration container."""
    io: IOConfig = Field(default_factory=IOConfig)
    realtime: RealtimeConfig = Field(default_factory=RealtimeConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)