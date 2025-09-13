"""Test configuration loading and validation."""

import pytest
from pathlib import Path
from privacy_redactor_rt.config import Config, IOConfig, RealtimeConfig


def test_default_config():
    """Test default configuration creation."""
    config = Config()
    assert config.io.target_width == 1280
    assert config.io.target_height == 720
    assert config.realtime.detector_stride == 3


def test_config_yaml_roundtrip(tmp_path):
    """Test configuration YAML serialization and deserialization."""
    config = Config()
    config.io.target_width = 1920
    config.realtime.detector_stride = 5
    
    yaml_path = tmp_path / "test_config.yaml"
    config.to_yaml(yaml_path)
    
    loaded_config = Config.from_yaml(yaml_path)
    assert loaded_config.io.target_width == 1920
    assert loaded_config.realtime.detector_stride == 5


def test_config_validation():
    """Test configuration validation."""
    # Test valid configuration
    config = Config(
        io=IOConfig(target_width=1280, target_height=720),
        realtime=RealtimeConfig(detector_stride=3)
    )
    assert config.io.target_width == 1280
    
    # Test invalid values would be caught by Pydantic
    with pytest.raises(ValueError):
        IOConfig(target_width="invalid")