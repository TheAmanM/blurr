"""Configuration models using Pydantic for validation and type safety."""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml
from pydantic import BaseModel, Field, field_validator


class IOConfig(BaseModel):
    """Input/Output configuration."""
    target_width: int = Field(default=1280, ge=320, le=3840)
    target_height: int = Field(default=720, ge=240, le=2160)
    target_fps: int = Field(default=30, ge=1, le=60)
    letterbox: bool = True
    letterbox_color: Tuple[int, int, int] = (0, 0, 0)

    @field_validator('letterbox_color')
    @classmethod
    def validate_color(cls, v):
        """Validate RGB color values."""
        if len(v) != 3 or not all(0 <= c <= 255 for c in v):
            raise ValueError("Color must be RGB tuple with values 0-255")
        return v


class RealtimeConfig(BaseModel):
    """Real-time processing configuration."""
    detector_stride: int = Field(default=3, ge=1, le=30)
    ocr_refresh_stride: int = Field(default=10, ge=1, le=100)
    max_parallel_ocr: int = Field(default=1, ge=1, le=8)
    max_queue: int = Field(default=2, ge=1, le=20)
    backpressure_threshold_ms: float = Field(default=120.0, ge=10.0, le=1000.0)
    min_fps_threshold: int = Field(default=24, ge=10, le=60)


class DetectionConfig(BaseModel):
    """Text detection configuration."""
    min_text_confidence: float = Field(default=0.6, ge=0.1, le=1.0)
    bbox_inflate_px: int = Field(default=6, ge=0, le=50)
    min_box_size: Tuple[int, int] = Field(default=(10, 10))
    max_box_size: Tuple[int, int] = Field(default=(800, 600))
    use_gpu: bool = False
    det_db_thresh: float = Field(default=0.3, ge=0.1, le=0.9)
    det_db_box_thresh: float = Field(default=0.6, ge=0.1, le=0.9)

    @field_validator('min_box_size', 'max_box_size')
    @classmethod
    def validate_box_size(cls, v):
        """Validate box size tuples."""
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError("Box size must be tuple of positive integers")
        return v


class OCRConfig(BaseModel):
    """OCR processing configuration."""
    min_ocr_confidence: float = Field(default=0.7, ge=0.1, le=1.0)
    use_gpu: bool = False
    rec_batch_num: int = Field(default=6, ge=1, le=20)
    max_text_length: int = Field(default=100, ge=10, le=1000)
    enable_mkldnn: bool = True
    cpu_threads: int = Field(default=4, ge=1, le=16)


class ClassificationConfig(BaseModel):
    """Classification engine configuration."""
    require_temporal_consensus: int = Field(default=2, ge=1, le=10)
    categories: List[str] = Field(default=[
        "phone", "credit_card", "email", "address", "api_key"
    ])
    entropy_threshold_bits_per_char: float = Field(default=3.5, ge=2.0, le=6.0)
    min_phone_digits: int = Field(default=10, ge=7, le=15)
    min_credit_card_digits: int = Field(default=13, ge=13, le=19)
    address_min_score: float = Field(default=0.6, ge=0.1, le=1.0)
    use_spacy_ner: bool = False
    spacy_model: str = "en_core_web_sm"

    @field_validator('categories')
    @classmethod
    def validate_categories(cls, v):
        """Validate supported categories."""
        valid_categories = {"phone", "credit_card", "email", "address", "api_key"}
        invalid = set(v) - valid_categories
        if invalid:
            raise ValueError(f"Invalid categories: {invalid}")
        return v


class TrackingConfig(BaseModel):
    """Optical flow tracking configuration."""
    iou_threshold: float = Field(default=0.5, ge=0.1, le=0.9)
    max_age: int = Field(default=30, ge=5, le=300)
    min_hits: int = Field(default=3, ge=1, le=10)
    smoothing_factor: float = Field(default=0.3, ge=0.0, le=1.0)
    max_flow_error: float = Field(default=50.0, ge=1.0, le=200.0)
    flow_quality_level: float = Field(default=0.01, ge=0.001, le=0.1)
    flow_min_distance: int = Field(default=10, ge=1, le=50)
    flow_block_size: int = Field(default=3, ge=3, le=15)


class RedactionConfig(BaseModel):
    """Redaction engine configuration."""
    default_method: str = Field(default="gaussian", pattern="^(gaussian|pixelate|solid)$")
    gaussian_kernel_size: int = Field(default=15, ge=3, le=51)
    gaussian_sigma: float = Field(default=5.0, ge=1.0, le=20.0)
    pixelate_block_size: int = Field(default=8, ge=2, le=32)
    solid_color: Tuple[int, int, int] = Field(default=(0, 0, 0))
    category_methods: Dict[str, str] = Field(default_factory=dict)
    inflate_bbox_px: int = Field(default=2, ge=0, le=20)

    @field_validator('gaussian_kernel_size')
    @classmethod
    def validate_kernel_size(cls, v):
        """Ensure kernel size is odd."""
        if v % 2 == 0:
            raise ValueError("Gaussian kernel size must be odd")
        return v

    @field_validator('solid_color')
    @classmethod
    def validate_solid_color(cls, v):
        """Validate RGB color values."""
        if len(v) != 3 or not all(0 <= c <= 255 for c in v):
            raise ValueError("Color must be RGB tuple with values 0-255")
        return v

    @field_validator('category_methods')
    @classmethod
    def validate_category_methods(cls, v):
        """Validate redaction methods per category."""
        valid_methods = {"gaussian", "pixelate", "solid"}
        invalid = set(v.values()) - valid_methods
        if invalid:
            raise ValueError(f"Invalid redaction methods: {invalid}")
        return v


class RecordingConfig(BaseModel):
    """Video recording configuration."""
    enabled: bool = False
    output_dir: str = "recordings"
    filename_template: str = "redacted_{timestamp}.mp4"
    codec: str = "libx264"
    crf: int = Field(default=23, ge=0, le=51)
    preset: str = Field(default="medium", pattern="^(ultrafast|superfast|veryfast|faster|fast|medium|slow|slower|veryslow)$")
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"
    constant_framerate: bool = True


class LoggingConfig(BaseModel):
    """Logging and audit configuration."""
    enabled: bool = True
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_file: Optional[str] = None
    audit_detections: bool = False
    audit_file: Optional[str] = None
    log_text_previews: bool = True
    max_preview_length: int = Field(default=50, ge=10, le=200)
    mask_text: bool = True
    mask_chars_visible: int = Field(default=3, ge=1, le=10)


class PerformanceConfig(BaseModel):
    """Performance monitoring and optimization."""
    enable_profiling: bool = False
    stats_window_size: int = Field(default=30, ge=10, le=300)
    memory_limit_mb: Optional[int] = Field(default=None, ge=100)
    auto_quality_scaling: bool = True
    quality_scale_factor: float = Field(default=0.8, ge=0.1, le=1.0)
    enable_memory_pooling: bool = True


class Config(BaseModel):
    """Main configuration container."""
    io: IOConfig = Field(default_factory=IOConfig)
    realtime: RealtimeConfig = Field(default_factory=RealtimeConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> 'Config':
        """Load configuration from YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tuples to lists for YAML serialization
        def convert_tuples(obj):
            if isinstance(obj, dict):
                return {k: convert_tuples(v) for k, v in obj.items()}
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, list):
                return [convert_tuples(item) for item in obj]
            else:
                return obj
        
        data = convert_tuples(self.model_dump())
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=False,
                indent=2
            )

    def merge_overrides(self, overrides: Dict[str, Any]) -> 'Config':
        """Create new config with overrides applied."""
        config_dict = self.model_dump()
        
        # Deep merge overrides
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged = deep_merge(config_dict, overrides)
        return self.__class__(**merged)


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration with fallback to default."""
    if config_path is None:
        config_path = Path("default.yaml")
    
    try:
        return Config.from_yaml(config_path)
    except FileNotFoundError:
        # Return default configuration if file not found
        return Config()
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}")