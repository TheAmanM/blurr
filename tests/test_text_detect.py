"""Simplified unit tests for text detection wrapper."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys

from privacy_redactor_rt.text_detect import TextDetector
from privacy_redactor_rt.config import DetectionConfig
from privacy_redactor_rt.types import BBox


class TestTextDetectorBasic:
    """Basic test cases for TextDetector class."""
    
    @pytest.fixture
    def config(self):
        """Default detection configuration for testing."""
        return DetectionConfig(
            min_text_confidence=0.6,
            bbox_inflate_px=6,
            min_box_size=(10, 10),
            max_box_size=(800, 600),
            use_gpu=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.6
        )
    
    @pytest.fixture
    def detector(self, config):
        """TextDetector instance for testing."""
        return TextDetector(config)
    
    def test_init(self, config):
        """Test TextDetector initialization."""
        detector = TextDetector(config)
        
        assert detector.config == config
        assert detector._detector is None
        assert detector._initialized is False
    
    def test_get_detector_info_before_init(self, detector):
        """Test detector info before initialization."""
        info = detector.get_detector_info()
        
        expected = {
            'initialized': False,
            'use_gpu': False,
            'min_confidence': 0.6,
            'min_box_size': (10, 10),
            'max_box_size': (800, 600),
            'bbox_inflate_px': 6,
            'det_db_thresh': 0.3,
            'det_db_box_thresh': 0.6
        }
        
        assert info == expected
    
    def test_detect_empty_frame(self, detector):
        """Test detection with empty frame."""
        empty_frame = np.array([])
        result = detector.detect(empty_frame)
        assert result == []
    
    def test_detect_none_frame(self, detector):
        """Test detection with None frame."""
        result = detector.detect(None)
        assert result == []
    
    def test_quad_to_bbox_valid(self, detector):
        """Test quadrilateral to bounding box conversion with valid input."""
        quad_points = [[10, 20], [100, 25], [95, 55], [5, 50]]
        
        bbox = detector._quad_to_bbox(quad_points)
        
        assert bbox is not None
        assert bbox.x1 == 5   # min x
        assert bbox.y1 == 20  # min y
        assert bbox.x2 == 100 # max x
        assert bbox.y2 == 55  # max y
        assert bbox.confidence == 0.6
    
    def test_quad_to_bbox_invalid_points(self, detector):
        """Test quadrilateral conversion with invalid points."""
        # Test various invalid inputs
        invalid_inputs = [
            [[10, 20], [10, 20], [10, 20], [10, 20]],  # Zero area
            [[100, 50], [10, 20]],  # Wrong number of points
            [["invalid", 20], [100, 25], [95, 55], [5, 50]],  # Non-numeric
            [],  # Empty list
        ]
        
        for invalid_input in invalid_inputs:
            bbox = detector._quad_to_bbox(invalid_input)
            assert bbox is None
    
    def test_is_valid_size(self, detector):
        """Test bounding box size validation."""
        # Valid size
        valid_bbox = BBox(x1=10, y1=10, x2=50, y2=40, confidence=0.8)
        assert detector._is_valid_size(valid_bbox) is True
        
        # Too small width
        small_width = BBox(x1=10, y1=10, x2=15, y2=40, confidence=0.8)
        assert detector._is_valid_size(small_width) is False
        
        # Too small height
        small_height = BBox(x1=10, y1=10, x2=50, y2=15, confidence=0.8)
        assert detector._is_valid_size(small_height) is False
        
        # Too large width
        large_width = BBox(x1=10, y1=10, x2=900, y2=40, confidence=0.8)
        assert detector._is_valid_size(large_width) is False
        
        # Too large height
        large_height = BBox(x1=10, y1=10, x2=50, y2=700, confidence=0.8)
        assert detector._is_valid_size(large_height) is False


class TestTextDetectorWithMocks:
    """Test cases that require mocking PaddleOCR."""
    
    @pytest.fixture
    def config(self):
        """Default detection configuration for testing."""
        return DetectionConfig(
            min_text_confidence=0.6,
            bbox_inflate_px=6,
            min_box_size=(10, 10),
            max_box_size=(800, 600),
            use_gpu=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.6
        )
    
    @pytest.fixture
    def detector(self, config):
        """TextDetector instance for testing."""
        return TextDetector(config)
    
    @pytest.fixture
    def sample_frame(self):
        """Sample BGR frame for testing."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_lazy_init_success(self, detector):
        """Test successful lazy initialization."""
        mock_detector_instance = Mock()
        mock_paddle_module = Mock()
        mock_paddle_module.PaddleOCR.return_value = mock_detector_instance
        
        with patch.dict('sys.modules', {'paddleocr': mock_paddle_module}):
            detector.lazy_init()
        
        assert detector._initialized is True
        assert detector._detector == mock_detector_instance
        
        # Verify PaddleOCR was called with correct parameters
        mock_paddle_module.PaddleOCR.assert_called_once_with(
            use_angle_cls=False,
            lang='en',
            det_model_dir=None,
            rec=False,
            use_gpu=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.6,
            show_log=False
        )
    
    def test_lazy_init_import_error(self, detector):
        """Test lazy_init with PaddleOCR import error."""
        # Simulate import error by not adding paddleocr to sys.modules
        with pytest.raises(RuntimeError, match="PaddleOCR not installed"):
            detector.lazy_init()
    
    def test_detect_valid_results(self, detector, sample_frame):
        """Test detection with valid text regions."""
        # Mock detection results - quadrilateral points
        mock_results = [[
            [[10, 20], [100, 20], [100, 50], [10, 50]],  # Valid rectangle
            [[200, 100], [300, 100], [300, 130], [200, 130]],  # Another valid rectangle
        ]]
        
        mock_detector_instance = Mock()
        mock_detector_instance.ocr.return_value = mock_results
        mock_paddle_module = Mock()
        mock_paddle_module.PaddleOCR.return_value = mock_detector_instance
        
        with patch.dict('sys.modules', {'paddleocr': mock_paddle_module}):
            result = detector.detect(sample_frame)
        
        assert len(result) == 2
        
        # Check first bounding box
        bbox1 = result[0]
        assert isinstance(bbox1, BBox)
        assert bbox1.x1 == 10
        assert bbox1.y1 == 20
        assert bbox1.x2 == 100
        assert bbox1.y2 == 50
        assert bbox1.confidence == 0.6  # Default confidence
        
        # Check second bounding box
        bbox2 = result[1]
        assert bbox2.x1 == 200
        assert bbox2.y1 == 100
        assert bbox2.x2 == 300
        assert bbox2.y2 == 130
    
    def test_detect_size_filtering(self, detector, sample_frame):
        """Test detection with size filtering."""
        # Mock results with various sizes
        mock_results = [[
            [[0, 0], [5, 0], [5, 5], [0, 5]],  # Too small (5x5, min is 10x10)
            [[0, 0], [1000, 0], [1000, 700], [0, 700]],  # Too large (1000x700, max is 800x600)
            [[10, 10], [50, 10], [50, 40], [10, 40]],  # Valid size (40x30)
        ]]
        
        mock_detector_instance = Mock()
        mock_detector_instance.ocr.return_value = mock_results
        mock_paddle_module = Mock()
        mock_paddle_module.PaddleOCR.return_value = mock_detector_instance
        
        with patch.dict('sys.modules', {'paddleocr': mock_paddle_module}):
            result = detector.detect(sample_frame)
        
        # Should only return the valid sized rectangle
        assert len(result) == 1
        assert result[0].width == 40
        assert result[0].height == 30