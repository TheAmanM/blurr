"""Simple integration tests for core functionality without heavy dependencies."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import threading
import time

from privacy_redactor_rt.config import Config


class TestConfigIntegration:
    """Test configuration system integration."""
    
    def test_config_creation_and_validation(self):
        """Test configuration creation with validation."""
        config = Config()
        
        # Test default values
        assert config.io.target_width == 1280
        assert config.io.target_height == 720
        assert config.io.target_fps == 30
        assert config.realtime.detector_stride == 3
        assert config.classification.categories == ["phone", "credit_card", "email", "address", "api_key"]
    
    def test_config_yaml_serialization(self):
        """Test YAML serialization and deserialization."""
        config = Config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            # Save to YAML
            config.to_yaml(config_path)
            assert config_path.exists()
            
            # Load from YAML
            loaded_config = Config.from_yaml(config_path)
            
            # Verify values match
            assert loaded_config.io.target_width == config.io.target_width
            assert loaded_config.realtime.detector_stride == config.realtime.detector_stride
            assert loaded_config.classification.categories == config.classification.categories
    
    def test_config_overrides(self):
        """Test configuration override merging."""
        config = Config()
        
        overrides = {
            'io': {
                'target_fps': 60
            },
            'classification': {
                'categories': ['phone', 'email']
            },
            'redaction': {
                'default_method': 'pixelate'
            }
        }
        
        merged_config = config.merge_overrides(overrides)
        
        # Verify overrides were applied
        assert merged_config.io.target_fps == 60
        assert merged_config.classification.categories == ['phone', 'email']
        assert merged_config.redaction.default_method == 'pixelate'
        
        # Verify other values remain unchanged
        assert merged_config.io.target_width == config.io.target_width
        assert merged_config.realtime.detector_stride == config.realtime.detector_stride
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = Config()
        assert config.io.target_width > 0
        assert config.io.target_height > 0
        assert config.io.target_fps > 0
        
        # Test invalid values raise validation errors
        with pytest.raises(Exception):
            Config(io={'target_width': -1})
        
        with pytest.raises(Exception):
            Config(classification={'categories': ['invalid_category']})


class TestThreadSafetyIntegration:
    """Test thread safety of core components."""
    
    def test_detection_counter_thread_safety(self):
        """Test thread-safe detection counter updates."""
        # Mock streamlit for this test
        with patch.dict('sys.modules', {'streamlit': Mock()}):
            # Import after mocking
            import sys
            sys.modules['streamlit'].session_state = {}
            
            # Create a simple counter class similar to the app
            class DetectionCounter:
                def __init__(self):
                    self.counters = {}
                    self._lock = threading.Lock()
                
                def add_event(self, category):
                    with self._lock:
                        self.counters[category] = self.counters.get(category, 0) + 1
            
            counter = DetectionCounter()
            
            def add_events(category, count):
                for i in range(count):
                    counter.add_event(category)
                    time.sleep(0.001)  # Small delay to encourage race conditions
            
            # Create multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=add_events, args=(f"category_{i}", 20))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Verify all events were counted correctly
            total_events = sum(counter.counters.values())
            assert total_events == 100  # 5 categories * 20 events each
            
            for i in range(5):
                assert counter.counters[f"category_{i}"] == 20


class TestErrorHandlingIntegration:
    """Test error handling and graceful degradation."""
    
    def test_config_loading_with_missing_file(self):
        """Test configuration loading with missing file."""
        from privacy_redactor_rt.config import load_config
        
        # Test with non-existent file
        config = load_config(Path("non_existent_config.yaml"))
        
        # Should return default configuration
        assert config is not None
        assert isinstance(config, Config)
        assert config.io.target_width == 1280  # Default value
    
    def test_config_loading_with_invalid_yaml(self):
        """Test configuration loading with invalid YAML."""
        from privacy_redactor_rt.config import Config
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid_config.yaml"
            
            # Write invalid YAML
            with open(config_path, 'w') as f:
                f.write("invalid: yaml: content: [")
            
            # Should raise an exception
            with pytest.raises(Exception):
                Config.from_yaml(config_path)
    
    def test_graceful_degradation_with_missing_dependencies(self):
        """Test graceful degradation when dependencies are missing."""
        # This test verifies that the code structure allows for graceful degradation
        # when optional dependencies are not available
        
        # Mock missing dependencies
        with patch.dict('sys.modules', {
            'paddleocr': None,
            'cv2': None,
            'streamlit_webrtc': None
        }):
            # Should still be able to create configuration
            config = Config()
            assert config is not None
            
            # Should still be able to work with basic data structures
            from privacy_redactor_rt.types import BBox, Detection
            
            bbox = BBox(x1=10, y1=20, x2=100, y2=200, confidence=0.8)
            assert bbox.x1 == 10
            assert bbox.confidence == 0.8
            
            detection = Detection(bbox=bbox, text="test", timestamp=123.45)
            assert detection.bbox == bbox
            assert detection.text == "test"


class TestDataModelIntegration:
    """Test data model integration and serialization."""
    
    def test_bbox_operations(self):
        """Test bounding box operations."""
        from privacy_redactor_rt.types import BBox
        
        bbox1 = BBox(x1=10, y1=20, x2=100, y2=200, confidence=0.8)
        bbox2 = BBox(x1=50, y1=60, x2=150, y2=250, confidence=0.9)
        
        # Test basic properties
        assert bbox1.width == 90
        assert bbox1.height == 180
        assert bbox1.area == 16200
        
        # Test IoU calculation
        iou = bbox1.iou(bbox2)
        assert 0 <= iou <= 1
        
        # Test inflation
        inflated = bbox1.inflate(5)
        assert inflated.x1 == 5
        assert inflated.y1 == 15
        assert inflated.x2 == 105
        assert inflated.y2 == 205
    
    def test_detection_and_match_integration(self):
        """Test detection and match data structures."""
        from privacy_redactor_rt.types import BBox, Detection, Match
        
        bbox = BBox(x1=10, y1=20, x2=100, y2=200, confidence=0.8)
        detection = Detection(bbox=bbox, text="555-123-4567", timestamp=123.45)
        
        match = Match(
            category="phone",
            confidence=0.95,
            masked_text="555-***-4567",
            bbox=bbox
        )
        
        # Test data integrity
        assert detection.bbox == bbox
        assert match.category == "phone"
        assert match.masked_text == "555-***-4567"
        
        # Test dataclass properties
        assert detection.text == "555-123-4567"
        assert detection.timestamp == 123.45
        
        assert match.category == "phone"
        assert match.confidence == 0.95


class TestPerformanceIntegration:
    """Test performance monitoring integration."""
    
    def test_statistics_collection(self):
        """Test statistics collection and aggregation."""
        # Create a simple statistics collector
        class StatsCollector:
            def __init__(self):
                self.stats = {
                    'frames_processed': 0,
                    'detections_count': 0,
                    'processing_times': []
                }
            
            def update_frame_stats(self, processing_time):
                self.stats['frames_processed'] += 1
                self.stats['processing_times'].append(processing_time)
            
            def update_detection_stats(self, count):
                self.stats['detections_count'] += count
            
            def get_summary(self):
                if self.stats['processing_times']:
                    avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
                else:
                    avg_time = 0
                
                return {
                    'frames_processed': self.stats['frames_processed'],
                    'detections_count': self.stats['detections_count'],
                    'avg_processing_time': avg_time
                }
        
        collector = StatsCollector()
        
        # Simulate processing
        for i in range(10):
            collector.update_frame_stats(0.05 + i * 0.01)  # Increasing processing time
            collector.update_detection_stats(i % 3)  # Variable detection count
        
        summary = collector.get_summary()
        
        assert summary['frames_processed'] == 10
        assert summary['detections_count'] == sum(i % 3 for i in range(10))
        assert 0.05 <= summary['avg_processing_time'] <= 0.15


if __name__ == "__main__":
    pytest.main([__file__])