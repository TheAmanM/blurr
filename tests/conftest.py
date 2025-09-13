"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
from privacy_redactor_rt.config import Config


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return Config()


@pytest.fixture
def config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_path = tmp_path / "test_config.yaml"
    config = Config()
    config.to_yaml(config_path)
    return config_path