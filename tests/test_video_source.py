"""Unit tests for video source normalization and frame processing."""

import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import cv2

from privacy_redactor_rt.video_source import (
    FrameNormalizer,
    FPSThrottler,
    VideoSource,
    convert_bgr_to_rgb,
    convert_rgb_to_bgr,
    preserve_aspect_ratio
)
from privacy_redactor_rt.config import IOConfig
from privacy_redactor_rt.types import BBox


class TestFrameNormalizer:
    """Test frame normalization with letterboxing."""
    
    def test_init(self):
        """Test FrameNormalizer initialization."""
        config = IOConfig(target_width=1280, target_height=720, letterbox=True)
        normalizer = FrameNormalizer(config)
        
        assert normalizer.target_width == 1280
        assert normalizer.target_height == 720
        assert normalizer.letterbox is True
        assert normalizer.letterbox_color == (0, 0, 0)
    
    def test_normalize_frame_invalid_input(self):
        """Test normalization with invalid input."""
        config = IOConfig()
        normalizer = FrameNormalizer(config)
        
        with pytest.raises(ValueError, match="Invalid input frame"):
            normalizer.normalize_frame(None)
        
        with pytest.raises(ValueError, match="Invalid input frame"):
            normalizer.normalize_frame(np.array([]))
    
    def test_normalize_frame_no_letterbox(self):
        """Test frame normalization without letterboxing."""
        config = IOConfig(target_width=640, target_height=480, letterbox=False)
        normalizer = FrameNormalizer(config)
        
        # Create test frame (800x600)
        frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        
        normalized, scale, offset = normalizer.normalize_frame(frame)
        
        assert normalized.shape == (480, 640, 3)
        assert scale == min(640/800, 480/600)  # 0.8
        assert offset == (0, 0)
    
    def test_normalize_frame_with_letterbox_wider_input(self):
        """Test normalization with letterboxing for wider input."""
        config = IOConfig(target_width=1280, target_height=720, letterbox=True)
        normalizer = FrameNormalizer(config)
        
        # Create wide test frame (1920x540) - wider than target aspect ratio
        frame = np.random.randint(0, 255, (540, 1920, 3), dtype=np.uint8)
        
        normalized, scale, offset = normalizer.normalize_frame(frame)
        
        assert normalized.shape == (720, 1280, 3)
        # Scale should be limited by width: 1280/1920 = 0.667
        expected_scale = 1280 / 1920
        assert abs(scale - expected_scale) < 0.001
        
        # New height after scaling: 540 * 0.667 = 360
        # Offset should center vertically: (720 - 360) / 2 = 180
        assert offset[0] == 0  # No horizontal offset
        assert offset[1] == (720 - int(540 * expected_scale)) // 2
    
    def test_normalize_frame_with_letterbox_taller_input(self):
        """Test normalization with letterboxing for taller input."""
        config = IOConfig(target_width=1280, target_height=720, letterbox=True)
        normalizer = FrameNormalizer(config)
        
        # Create tall test frame (960x1080) - taller than target aspect ratio
        frame = np.random.randint(0, 255, (1080, 960, 3), dtype=np.uint8)
        
        normalized, scale, offset = normalizer.normalize_frame(frame)
        
        assert normalized.shape == (720, 1280, 3)
        # Scale should be limited by height: 720/1080 = 0.667
        expected_scale = 720 / 1080
        assert abs(scale - expected_scale) < 0.001
        
        # New width after scaling: 960 * 0.667 = 640
        # Offset should center horizontally: (1280 - 640) / 2 = 320
        assert offset[0] == (1280 - int(960 * expected_scale)) // 2
        assert offset[1] == 0  # No vertical offset
    
    def test_normalize_frame_exact_aspect_ratio(self):
        """Test normalization when input has exact target aspect ratio."""
        config = IOConfig(target_width=1280, target_height=720, letterbox=True)
        normalizer = FrameNormalizer(config)
        
        # Create frame with exact aspect ratio (640x360 = 16:9)
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        
        normalized, scale, offset = normalizer.normalize_frame(frame)
        
        assert normalized.shape == (720, 1280, 3)
        assert scale == 2.0  # 1280/640 = 720/360 = 2.0
        assert offset == (0, 0)  # No letterboxing needed
    
    def test_letterbox_color_custom(self):
        """Test custom letterbox color."""
        config = IOConfig(
            target_width=1280,
            target_height=720,
            letterbox=True,
            letterbox_color=(128, 64, 192)
        )
        normalizer = FrameNormalizer(config)
        
        # Create small frame to force letterboxing
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        normalized, scale, offset = normalizer.normalize_frame(frame)
        
        # Check that letterbox areas have the custom color
        # Top letterbox area
        top_color = normalized[0, 0]
        assert tuple(top_color) == (128, 64, 192)
        
        # Bottom letterbox area
        bottom_color = normalized[-1, 0]
        assert tuple(bottom_color) == (128, 64, 192)
    
    def test_denormalize_bbox(self):
        """Test bounding box denormalization."""
        config = IOConfig(target_width=1280, target_height=720, letterbox=True)
        normalizer = FrameNormalizer(config)
        
        # Test with known scale and offset
        scale = 0.5
        offset = (100, 50)
        
        # Normalized bbox
        bbox = BBox(x1=150, y1=100, x2=250, y2=200, confidence=0.8)
        
        denormalized = normalizer.denormalize_bbox(bbox, scale, offset)
        
        # Expected: remove offset then scale back
        # x1: (150 - 100) / 0.5 = 100
        # y1: (100 - 50) / 0.5 = 100
        # x2: (250 - 100) / 0.5 = 300
        # y2: (200 - 50) / 0.5 = 300
        assert denormalized.x1 == 100
        assert denormalized.y1 == 100
        assert denormalized.x2 == 300
        assert denormalized.y2 == 300
        assert denormalized.confidence == 0.8
    
    def test_normalize_bbox(self):
        """Test bounding box normalization."""
        config = IOConfig(target_width=1280, target_height=720, letterbox=True)
        normalizer = FrameNormalizer(config)
        
        # Test with known scale and offset
        scale = 0.5
        offset = (100, 50)
        
        # Original bbox
        bbox = BBox(x1=100, y1=100, x2=300, y2=300, confidence=0.8)
        
        normalized = normalizer.normalize_bbox(bbox, scale, offset)
        
        # Expected: scale then add offset
        # x1: 100 * 0.5 + 100 = 150
        # y1: 100 * 0.5 + 50 = 100
        # x2: 300 * 0.5 + 100 = 250
        # y2: 300 * 0.5 + 50 = 200
        assert normalized.x1 == 150
        assert normalized.y1 == 100
        assert normalized.x2 == 250
        assert normalized.y2 == 200
        assert normalized.confidence == 0.8
    
    def test_bbox_roundtrip(self):
        """Test that bbox normalization and denormalization are inverse operations."""
        config = IOConfig(target_width=1280, target_height=720, letterbox=True)
        normalizer = FrameNormalizer(config)
        
        scale = 0.75
        offset = (64, 32)
        
        original_bbox = BBox(x1=50, y1=100, x2=200, y2=250, confidence=0.9)
        
        # Normalize then denormalize
        normalized = normalizer.normalize_bbox(original_bbox, scale, offset)
        denormalized = normalizer.denormalize_bbox(normalized, scale, offset)
        
        # Should get back original coordinates (within rounding error)
        assert abs(denormalized.x1 - original_bbox.x1) <= 1
        assert abs(denormalized.y1 - original_bbox.y1) <= 1
        assert abs(denormalized.x2 - original_bbox.x2) <= 1
        assert abs(denormalized.y2 - original_bbox.y2) <= 1
        assert denormalized.confidence == original_bbox.confidence


class TestFPSThrottler:
    """Test FPS throttling functionality."""
    
    def test_init(self):
        """Test FPSThrottler initialization."""
        throttler = FPSThrottler(30)
        
        assert throttler.target_fps == 30
        assert throttler.frame_interval == 1.0 / 30
        assert throttler.frame_count == 0
    
    def test_should_process_frame_initial(self):
        """Test first frame should always be processed."""
        throttler = FPSThrottler(30)
        
        assert throttler.should_process_frame() is True
        assert throttler.frame_count == 1
    
    def test_should_process_frame_timing(self):
        """Test frame processing timing."""
        throttler = FPSThrottler(10)  # 10 FPS = 0.1s interval
        
        # First frame should be processed
        assert throttler.should_process_frame() is True
        
        # Immediate second call should be skipped
        assert throttler.should_process_frame() is False
        
        # After waiting, should process again
        time.sleep(0.11)  # Wait longer than interval
        assert throttler.should_process_frame() is True
    
    def test_wait_for_next_frame(self):
        """Test waiting for next frame."""
        throttler = FPSThrottler(10)  # 10 FPS = 0.1s interval
        
        # First call sets the baseline
        throttler.wait_for_next_frame()
        assert throttler.frame_count == 1
        
        # Second call should wait for the interval
        start_time = time.time()
        throttler.wait_for_next_frame()
        elapsed = time.time() - start_time
        
        # Should have waited approximately the frame interval
        assert elapsed >= 0.09  # Allow some tolerance
        assert throttler.frame_count == 2
    
    def test_get_actual_fps_no_frames(self):
        """Test FPS calculation with no frames."""
        throttler = FPSThrottler(30)
        
        assert throttler.get_actual_fps() == 0.0
    
    def test_get_actual_fps_with_frames(self):
        """Test FPS calculation with processed frames."""
        throttler = FPSThrottler(30)
        
        # Process some frames with known timing
        throttler.should_process_frame()
        time.sleep(0.1)
        throttler.should_process_frame()
        
        fps = throttler.get_actual_fps()
        assert fps > 0
        # Should be roughly 10 FPS (2 frames in ~0.1s)
        assert 5 < fps < 25  # Allow wide tolerance for test timing
    
    def test_reset(self):
        """Test throttler reset."""
        throttler = FPSThrottler(30)
        
        # Process some frames
        throttler.should_process_frame()
        throttler.should_process_frame()
        
        assert throttler.frame_count > 0
        
        # Reset
        throttler.reset()
        
        assert throttler.frame_count == 0
        assert throttler.get_actual_fps() == 0.0


class TestVideoSource:
    """Test unified video source interface."""
    
    def test_init(self):
        """Test VideoSource initialization."""
        config = IOConfig()
        source = VideoSource(config)
        
        assert source.config == config
        assert isinstance(source.normalizer, FrameNormalizer)
        assert isinstance(source.fps_throttler, FPSThrottler)
        assert source.cap is None
    
    @patch('cv2.VideoCapture')
    def test_open_webcam_success(self, mock_capture):
        """Test successful webcam opening."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_capture.return_value = mock_cap
        
        config = IOConfig()
        source = VideoSource(config)
        
        result = source.open_webcam(0)
        
        assert result is True
        assert source.cap == mock_cap
        assert source.source_fps == 30.0
        assert source.total_frames is None
        
        # Verify webcam configuration calls
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FPS, 30)
    
    @patch('cv2.VideoCapture')
    def test_open_webcam_failure(self, mock_capture):
        """Test webcam opening failure."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap
        
        config = IOConfig()
        source = VideoSource(config)
        
        result = source.open_webcam(0)
        
        assert result is False
    
    @patch('cv2.VideoCapture')
    def test_open_rtsp_success(self, mock_capture):
        """Test successful RTSP stream opening."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0
        mock_capture.return_value = mock_cap
        
        config = IOConfig()
        source = VideoSource(config)
        
        result = source.open_rtsp("rtsp://example.com/stream")
        
        assert result is True
        assert source.cap == mock_cap
        assert source.source_fps == 25.0
        assert source.total_frames is None
        
        # Verify RTSP configuration
        mock_cap.set.assert_called_with(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    @patch('cv2.VideoCapture')
    def test_open_rtsp_invalid_fps(self, mock_capture):
        """Test RTSP with invalid FPS fallback."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0.0  # Invalid FPS
        mock_capture.return_value = mock_cap
        
        config = IOConfig()
        source = VideoSource(config)
        
        result = source.open_rtsp("rtsp://example.com/stream")
        
        assert result is True
        assert source.source_fps == 30.0  # Fallback value
    
    @patch('cv2.VideoCapture')
    def test_open_file_success(self, mock_capture):
        """Test successful file opening."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 24.0,
            cv2.CAP_PROP_FRAME_COUNT: 1000
        }.get(prop, 0)
        mock_capture.return_value = mock_cap
        
        config = IOConfig()
        source = VideoSource(config)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            result = source.open_file(tmp_path)
            
            assert result is True
            assert source.cap == mock_cap
            assert source.source_fps == 24.0
            assert source.total_frames == 1000
        finally:
            tmp_path.unlink()
    
    def test_open_file_not_exists(self):
        """Test opening non-existent file."""
        config = IOConfig()
        source = VideoSource(config)
        
        result = source.open_file("nonexistent.mp4")
        
        assert result is False
    
    @patch('cv2.VideoCapture')
    def test_read_frame_success(self, mock_capture):
        """Test successful frame reading."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        
        config = IOConfig(target_width=320, target_height=240)
        source = VideoSource(config)
        source.cap = mock_cap
        
        result = source.read_frame()
        
        assert result is not None
        normalized_frame, scale, offset = result
        assert normalized_frame.shape == (240, 320, 3)
        assert isinstance(scale, float)
        assert isinstance(offset, tuple)
        assert source.current_frame == 1
    
    @patch('cv2.VideoCapture')
    def test_read_frame_failure(self, mock_capture):
        """Test frame reading failure."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        
        config = IOConfig()
        source = VideoSource(config)
        source.cap = mock_cap
        
        result = source.read_frame()
        
        assert result is None
    
    def test_read_frame_no_capture(self):
        """Test reading frame without opened capture."""
        config = IOConfig()
        source = VideoSource(config)
        
        result = source.read_frame()
        
        assert result is None
    
    @patch('cv2.VideoCapture')
    def test_read_frame_throttled_live_source(self, mock_capture):
        """Test throttled reading for live sources (should not throttle)."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        
        config = IOConfig()
        source = VideoSource(config)
        source.cap = mock_cap
        source.total_frames = None  # Live source
        
        # Should always return frame for live sources
        result1 = source.read_frame_throttled()
        result2 = source.read_frame_throttled()
        
        assert result1 is not None
        assert result2 is not None
    
    @patch('cv2.VideoCapture')
    def test_read_frame_throttled_file_source(self, mock_capture):
        """Test throttled reading for file sources."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        
        config = IOConfig(target_fps=60)  # High FPS for testing (within validation limits)
        source = VideoSource(config)
        source.cap = mock_cap
        source.total_frames = 100  # File source
        
        # First call should return frame
        result1 = source.read_frame_throttled()
        assert result1 is not None
        
        # Immediate second call should be throttled (return None)
        result2 = source.read_frame_throttled()
        assert result2 is None
    
    def test_get_frame_iterator(self):
        """Test frame iterator."""
        config = IOConfig()
        source = VideoSource(config)
        
        # Mock read_frame_throttled to return frames then None
        frames = [
            (np.zeros((720, 1280, 3)), 1.0, (0, 0)),
            (np.ones((720, 1280, 3)), 1.0, (0, 0)),
            None  # End of stream
        ]
        
        def mock_read():
            return frames.pop(0) if frames else None
        
        source.read_frame_throttled = mock_read
        
        frame_list = list(source.get_frame_iterator())
        
        assert len(frame_list) == 2
        assert frame_list[0][0].shape == (720, 1280, 3)
        assert frame_list[1][0].shape == (720, 1280, 3)
    
    def test_get_progress_live_source(self):
        """Test progress for live sources."""
        config = IOConfig()
        source = VideoSource(config)
        source.total_frames = None  # Live source
        
        progress = source.get_progress()
        
        assert progress is None
    
    def test_get_progress_file_source(self):
        """Test progress for file sources."""
        config = IOConfig()
        source = VideoSource(config)
        source.total_frames = 100
        source.current_frame = 25
        
        progress = source.get_progress()
        
        assert progress == 0.25
    
    def test_get_progress_complete(self):
        """Test progress when file is complete."""
        config = IOConfig()
        source = VideoSource(config)
        source.total_frames = 100
        source.current_frame = 150  # Beyond total
        
        progress = source.get_progress()
        
        assert progress == 1.0
    
    def test_get_source_info(self):
        """Test source information retrieval."""
        config = IOConfig(target_fps=30, target_width=1280, target_height=720)
        source = VideoSource(config)
        
        # Mock a capture object to make get_source_info work
        mock_cap = Mock()
        source.cap = mock_cap
        source.source_fps = 25.0
        source.total_frames = 1000
        source.current_frame = 250
        
        info = source.get_source_info()
        
        assert info["source_fps"] == 25.0
        assert info["total_frames"] == 1000
        assert info["current_frame"] == 250
        assert info["progress"] == 0.25
        assert info["target_fps"] == 30
        assert info["target_resolution"] == (1280, 720)
        assert "actual_fps" in info
    
    def test_get_source_info_no_capture(self):
        """Test source info with no capture."""
        config = IOConfig()
        source = VideoSource(config)
        
        info = source.get_source_info()
        
        assert info == {}
    
    def test_close(self):
        """Test closing video source."""
        config = IOConfig()
        source = VideoSource(config)
        
        mock_cap = Mock()
        source.cap = mock_cap
        source.current_frame = 100
        
        source.close()
        
        mock_cap.release.assert_called_once()
        assert source.cap is None
        assert source.current_frame == 0


class TestUtilityFunctions:
    """Test utility functions for frame conversion."""
    
    def test_convert_bgr_to_rgb(self):
        """Test BGR to RGB conversion."""
        # Create test BGR frame
        bgr_frame = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Blue in BGR
        
        rgb_frame = convert_bgr_to_rgb(bgr_frame)
        
        # Should be red in RGB
        assert rgb_frame[0, 0, 0] == 0    # R
        assert rgb_frame[0, 0, 1] == 0    # G
        assert rgb_frame[0, 0, 2] == 255  # B
    
    def test_convert_rgb_to_bgr(self):
        """Test RGB to BGR conversion."""
        # Create test RGB frame
        rgb_frame = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Red in RGB
        
        bgr_frame = convert_rgb_to_bgr(rgb_frame)
        
        # Should be blue in BGR
        assert bgr_frame[0, 0, 0] == 0    # B
        assert bgr_frame[0, 0, 1] == 0    # G
        assert bgr_frame[0, 0, 2] == 255  # R
    
    def test_color_conversion_roundtrip(self):
        """Test that color conversions are inverse operations."""
        original = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # BGR -> RGB -> BGR
        converted = convert_rgb_to_bgr(convert_bgr_to_rgb(original))
        
        np.testing.assert_array_equal(original, converted)
    
    def test_preserve_aspect_ratio_wider_target(self):
        """Test aspect ratio preservation when target is wider."""
        new_width, new_height, scale = preserve_aspect_ratio(
            original_width=800,
            original_height=600,
            target_width=1600,
            target_height=900
        )
        
        # Should be limited by height: 900/600 = 1.5
        assert scale == 1.5
        assert new_width == 1200  # 800 * 1.5
        assert new_height == 900  # 600 * 1.5
    
    def test_preserve_aspect_ratio_taller_target(self):
        """Test aspect ratio preservation when target is taller."""
        new_width, new_height, scale = preserve_aspect_ratio(
            original_width=1920,
            original_height=1080,
            target_width=1280,
            target_height=1440
        )
        
        # Should be limited by width: 1280/1920 = 0.667
        expected_scale = 1280 / 1920
        assert abs(scale - expected_scale) < 0.001
        assert new_width == 1280
        assert new_height == int(1080 * expected_scale)
    
    def test_preserve_aspect_ratio_exact_match(self):
        """Test aspect ratio preservation with exact match."""
        new_width, new_height, scale = preserve_aspect_ratio(
            original_width=640,
            original_height=360,
            target_width=1280,
            target_height=720
        )
        
        # Exact 16:9 aspect ratio match
        assert scale == 2.0
        assert new_width == 1280
        assert new_height == 720


if __name__ == "__main__":
    pytest.main([__file__])