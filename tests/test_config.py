"""Unit tests for configuration system."""

import pytest
import tempfile
from pathlib import Path
from pydantic import ValidationError
import yaml

from privacy_redactor_rt.config import (
    Config, IOConfig, RealtimeConfig, DetectionConfig, OCRConfig,
    ClassificationConfig, TrackingConfig, RedactionConfig,
    RecordingConfig, LoggingConfig, PerformanceConfig, load_config
)


class TestIOConfig:
    """Test IOConfig validation."""

    def test_valid_config(self):
        """Test valid IO configuration."""
        config = IOConfig(
            target_width=1920,
            target_height=1080,
            target_fps=30,
            letterbox=True,
            letterbox_color=(128, 128, 128)
        )
        assert config.target_width == 1920
        assert config.target_height == 1080
        assert config.letterbox_color == (128, 128, 128)

    def test_invalid_dimensions(self):
        """Test invalid dimension validation."""
        with pytest.raises(ValidationError):
            IOConfig(target_width=100)  # Too small
        
        with pytest.raises(ValidationError):
            IOConfig(target_height=5000)  # Too large

    def test_invalid_color(self):
        """Test invalid color validation."""
        with pytest.raises(ValidationError):
            IOConfig(letterbox_color=(256, 0, 0))  # Value too high
        
        with pytest.raises(ValidationError):
            IOConfig(letterbox_color=(0, 0))  # Wrong length


class TestRealtimeConfig:
    """Test RealtimeConfig validation."""

    def test_valid_config(self):
        """Test valid realtime configuration."""
        config = RealtimeConfig(
            detector_stride=5,
            ocr_refresh_stride=15,
            max_parallel_ocr=2,
            max_queue=5
        )
        assert config.detector_stride == 5
        assert config.ocr_refresh_stride == 15

    def test_invalid_values(self):
        """Test invalid value validation."""
        with pytest.raises(ValidationError):
            RealtimeConfig(detector_stride=0)  # Too small
        
        with pytest.raises(ValidationError):
            RealtimeConfig(max_parallel_ocr=10)  # Too large


class TestDetectionConfig:
    """Test DetectionConfig validation."""

    def test_valid_config(self):
        """Test valid detection configuration."""
        config = DetectionConfig(
            min_text_confidence=0.8,
            bbox_inflate_px=10,
            min_box_size=(15, 15),
            max_box_size=(1000, 800)
        )
        assert config.min_text_confidence == 0.8
        assert config.min_box_size == (15, 15)

    def test_invalid_box_size(self):
        """Test invalid box size validation."""
        with pytest.raises(ValidationError):
            DetectionConfig(min_box_size=(0, 10))  # Zero width
        
        with pytest.raises(ValidationError):
            DetectionConfig(max_box_size=(10,))  # Wrong length


class TestClassificationConfig:
    """Test ClassificationConfig validation."""

    def test_valid_config(self):
        """Test valid classification configuration."""
        config = ClassificationConfig(
            categories=["phone", "email", "credit_card"],
            entropy_threshold_bits_per_char=4.0
        )
        assert "phone" in config.categories
        assert config.entropy_threshold_bits_per_char == 4.0

    def test_invalid_categories(self):
        """Test invalid category validation."""
        with pytest.raises(ValidationError):
            ClassificationConfig(categories=["phone", "invalid_category"])


class TestRedactionConfig:
    """Test RedactionConfig validation."""

    def test_valid_config(self):
        """Test valid redaction configuration."""
        config = RedactionConfig(
            default_method="pixelate",
            gaussian_kernel_size=21,
            category_methods={"phone": "gaussian", "email": "solid"}
        )
        assert config.default_method == "pixelate"
        assert config.gaussian_kernel_size == 21

    def test_invalid_method(self):
        """Test invalid method validation."""
        with pytest.raises(ValidationError):
            RedactionConfig(default_method="invalid_method")

    def test_invalid_kernel_size(self):
        """Test invalid kernel size validation."""
        with pytest.raises(ValidationError):
            RedactionConfig(gaussian_kernel_size=20)  # Must be odd

    def test_invalid_category_methods(self):
        """Test invalid category methods validation."""
        with pytest.raises(ValidationError):
            RedactionConfig(category_methods={"phone": "invalid_method"})


class TestRecordingConfig:
    """Test RecordingConfig validation."""

    def test_valid_config(self):
        """Test valid recording configuration."""
        config = RecordingConfig(
            enabled=True,
            codec="libx264",
            crf=18,
            preset="fast"
        )
        assert config.enabled is True
        assert config.crf == 18

    def test_invalid_preset(self):
        """Test invalid preset validation."""
        with pytest.raises(ValidationError):
            RecordingConfig(preset="invalid_preset")

    def test_invalid_crf(self):
        """Test invalid CRF validation."""
        with pytest.raises(ValidationError):
            RecordingConfig(crf=60)  # Too high


class TestLoggingConfig:
    """Test LoggingConfig validation."""

    def test_valid_config(self):
        """Test valid logging configuration."""
        config = LoggingConfig(
            log_level="DEBUG",
            mask_chars_visible=5
        )
        assert config.log_level == "DEBUG"
        assert config.mask_chars_visible == 5

    def test_invalid_log_level(self):
        """Test invalid log level validation."""
        with pytest.raises(ValidationError):
            LoggingConfig(log_level="INVALID")


class TestConfig:
    """Test main Config class."""

    def test_default_config(self):
        """Test creating default configuration."""
        config = Config()
        assert isinstance(config.io, IOConfig)
        assert isinstance(config.realtime, RealtimeConfig)
        assert isinstance(config.detection, DetectionConfig)
        assert config.io.target_width == 1280
        assert config.realtime.detector_stride == 3

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "io": {"target_width": 1920, "target_height": 1080},
            "realtime": {"detector_stride": 5},
            "detection": {"min_text_confidence": 0.8}
        }
        
        config = Config.from_dict(data)
        assert config.io.target_width == 1920
        assert config.io.target_height == 1080
        assert config.realtime.detector_stride == 5
        assert config.detection.min_text_confidence == 0.8

    def test_config_merge_overrides(self):
        """Test merging configuration overrides."""
        config = Config()
        original_width = config.io.target_width
        
        overrides = {
            "io": {"target_width": 1920},
            "realtime": {"detector_stride": 5}
        }
        
        merged = config.merge_overrides(overrides)
        
        # Original should be unchanged
        assert config.io.target_width == original_width
        
        # Merged should have overrides
        assert merged.io.target_width == 1920
        assert merged.realtime.detector_stride == 5
        
        # Other values should remain default
        assert merged.io.target_height == config.io.target_height

    def test_config_yaml_roundtrip(self):
        """Test saving and loading YAML configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            
            # Create config with custom values
            config = Config()
            config.io.target_width = 1920
            config.realtime.detector_stride = 5
            
            # Save to YAML
            config.to_yaml(config_path)
            assert config_path.exists()
            
            # Load from YAML
            loaded_config = Config.from_yaml(config_path)
            
            assert loaded_config.io.target_width == 1920
            assert loaded_config.realtime.detector_stride == 5

    def test_config_yaml_validation(self):
        """Test YAML configuration validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid_config.yaml"
            
            # Create invalid YAML
            invalid_data = {
                "io": {"target_width": 100},  # Too small
                "detection": {"min_text_confidence": 2.0}  # Too high
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(invalid_data, f)
            
            with pytest.raises(ValidationError):
                Config.from_yaml(config_path)

    def test_config_missing_file(self):
        """Test handling missing configuration file."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml(Path("nonexistent.yaml"))


class TestLoadConfig:
    """Test load_config function."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        # Should return default config when no file exists
        config = load_config(Path("nonexistent.yaml"))
        assert isinstance(config, Config)
        assert config.io.target_width == 1280

    def test_load_existing_config(self):
        """Test loading existing configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            
            # Create test config
            test_data = {
                "io": {"target_width": 1920},
                "realtime": {"detector_stride": 7}
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(test_data, f)
            
            config = load_config(config_path)
            assert config.io.target_width == 1920
            assert config.realtime.detector_stride == 7

    def test_load_config_validation_error(self):
        """Test handling validation errors in config loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid_config.yaml"
            
            # Create invalid config
            invalid_data = {"io": {"target_width": "invalid"}}
            
            with open(config_path, 'w') as f:
                yaml.dump(invalid_data, f)
            
            with pytest.raises(ValueError, match="Failed to load configuration"):
                load_config(config_path)

    def test_load_config_none_path(self):
        """Test loading config with None path."""
        # Should try to load default.yaml, fall back to default config
        config = load_config(None)
        assert isinstance(config, Config)