"""Integration tests for WebRTC frame handling."""

import time
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch
import av

from privacy_redactor_rt.webrtc_utils import VideoTransformer
from privacy_redactor_rt.config import Config
from privacy_redactor_rt.pipeline import RealtimePipeline
from privacy_redactor_rt.types import BBox, Detection, Match, Track


class TestVideoTransformer:
    """Test WebRTC VideoTransformer functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline for testing."""
        pipeline = Mock(spec=RealtimePipeline)
        pipeline.process_frame.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)
        pipeline.get_stats.return_value = {
            'frames_processed': 100,
            'detections_run': 33,
            'ocr_requests': 15,
            'tracks_active': 2,
            'avg_processing_time': 0.05,
            'ocr_queue_size': 1,
            'ocr_cache_size': 5,
            'ocr_processed': 15,
            'ocr_queue_full': 0,
            'consensus_buffer_size': 2,
            'total_tracks': 3
        }
        return pipeline
    
    @pytest.fixture
    def transformer(self, config, mock_pipeline):
        """Create VideoTransformer instance."""
        return VideoTransformer(config, mock_pipeline)
    
    @pytest.fixture
    def mock_frame(self):
        """Create mock av.VideoFrame."""
        frame = Mock(spec=av.VideoFrame)
        frame.to_ndarray.return_value = np.random.randint(
            0, 255, (480, 640, 3), dtype=np.uint8
        )
        frame.pts = 1000
        frame.time_base = av.Rational(1, 30)
        return frame
    
    def test_transformer_initialization(self, config, mock_pipeline):
        """Test VideoTransformer initialization."""
        transformer = VideoTransformer(config, mock_pipeline)
        
        assert transformer.config == config
        assert transformer.pipeline == mock_pipeline
        assert transformer.frame_count == 0
        assert transformer.dropped_frames == 0
        assert transformer.total_frames == 0
        assert transformer.fps_ema == 0.0
        assert transformer.latency_ema == 0.0
        assert transformer.processing_ema == 0.0
    
    def test_frame_processing_basic(self, transformer, mock_frame):
        """Test basic frame processing."""
        with patch('av.VideoFrame.from_ndarray') as mock_from_ndarray:
            mock_output_frame = Mock(spec=av.VideoFrame)
            mock_from_ndarray.return_value = mock_output_frame
            
            result = transformer.recv(mock_frame)
            
            # Verify pipeline was called
            transformer.pipeline.process_frame.assert_called_once()
            
            # Verify frame conversion
            mock_from_ndarray.assert_called_once()
            
            # Verify frame properties are preserved
            assert mock_output_frame.pts == mock_frame.pts
            assert mock_output_frame.time_base == mock_frame.time_base
            
            # Verify frame count incremented
            assert transformer.frame_count == 1
            assert transformer.total_frames == 1
    
    def test_frame_normalization(self, transformer):
        """Test frame normalization to target resolution."""
        # Test with different input sizes
        test_cases = [
            (640, 480),   # 4:3 aspect ratio
            (1920, 1080), # 16:9 aspect ratio
            (800, 600),   # 4:3 smaller
            (320, 240),   # Very small
        ]
        
        for width, height in test_cases:
            img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            normalized = transformer._normalize_frame(img)
            
            # Should always be target resolution
            assert normalized.shape == (720, 1280, 3)
            assert normalized.dtype == np.uint8
    
    def test_letterbox_caching(self, transformer):
        """Test letterbox parameter caching."""
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # First call should create cache
        transformer._normalize_frame(img1)
        cache1 = transformer._letterbox_cache.copy()
        
        # Second call with same dimensions should use cache
        transformer._normalize_frame(img2)
        cache2 = transformer._letterbox_cache
        
        assert cache1 == cache2
        
        # Different dimensions should update cache
        img3 = np.random.randint(0, 255, (720, 960, 3), dtype=np.uint8)
        transformer._normalize_frame(img3)
        cache3 = transformer._letterbox_cache
        
        assert cache1 != cache3
    
    def test_performance_stats_update(self, transformer, mock_frame):
        """Test performance statistics updates."""
        # Process several frames
        with patch('av.VideoFrame.from_ndarray'):
            for i in range(5):
                transformer.recv(mock_frame)
                time.sleep(0.01)  # Small delay to create measurable intervals
        
        stats = transformer.get_stats()
        
        # Check basic counters
        assert stats['frames_total'] == 5
        assert stats['frames_processed'] == 5
        assert stats['frames_dropped'] == 0
        
        # Check EMA values are updated
        assert stats['fps_current'] > 0
        assert stats['latency_current_ms'] > 0
        assert stats['processing_current_ms'] >= 0
        
        # Check averages are calculated
        assert stats['fps_avg'] > 0
        assert stats['latency_avg_ms'] > 0
        assert stats['processing_avg_ms'] >= 0
    
    def test_backpressure_frame_dropping(self, transformer, mock_frame):
        """Test frame dropping under backpressure."""
        # Mock slow processing
        transformer.pipeline.process_frame.side_effect = lambda *args: (
            time.sleep(0.2),  # Simulate slow processing
            np.zeros((720, 1280, 3), dtype=np.uint8)
        )[1]
        
        with patch('av.VideoFrame.from_ndarray'):
            # Process frames rapidly
            for i in range(10):
                transformer.recv(mock_frame)
        
        stats = transformer.get_stats()
        
        # Should have dropped some frames due to slow processing
        assert stats['frames_dropped'] > 0
        assert stats['drop_rate_percent'] > 0
    
    def test_fps_threshold_dropping(self, transformer, mock_frame):
        """Test frame dropping when FPS falls below threshold."""
        # Set low FPS threshold
        transformer.config.realtime.min_fps_threshold = 25
        
        # Simulate consistently low FPS
        transformer.fps_ema = 20.0  # Below threshold
        
        with patch('av.VideoFrame.from_ndarray'), \
             patch('numpy.random.random', return_value=0.1):  # Force drop
            
            dropped_before = transformer.dropped_frames
            transformer.recv(mock_frame)
            
            # Should drop frame due to low FPS
            assert transformer.dropped_frames > dropped_before
    
    def test_error_handling(self, transformer, mock_frame):
        """Test error handling in frame processing."""
        # Mock pipeline to raise exception
        transformer.pipeline.process_frame.side_effect = Exception("Test error")
        
        # Should return original frame on error
        result = transformer.recv(mock_frame)
        assert result == mock_frame
        
        # Error should be logged but not crash
        assert transformer.frame_count == 0  # Frame not counted due to error
    
    def test_fallback_frame_creation(self, transformer, mock_frame):
        """Test fallback frame creation."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with patch('av.VideoFrame.from_ndarray') as mock_from_ndarray:
            mock_fallback = Mock(spec=av.VideoFrame)
            mock_from_ndarray.return_value = mock_fallback
            
            result = transformer._create_fallback_frame(mock_frame, img)
            
            # Should create black frame
            mock_from_ndarray.assert_called_once()
            call_args = mock_from_ndarray.call_args[0]
            black_img = call_args[0]
            
            assert np.all(black_img == 0)  # Should be all black
            assert black_img.shape == img.shape
            
            # Frame properties should be preserved
            assert mock_fallback.pts == mock_frame.pts
            assert mock_fallback.time_base == mock_frame.time_base
    
    def test_stats_comprehensive(self, transformer, mock_frame):
        """Test comprehensive statistics collection."""
        with patch('av.VideoFrame.from_ndarray'):
            # Process frames with varying delays
            for i in range(10):
                transformer.recv(mock_frame)
                if i % 3 == 0:
                    time.sleep(0.01)  # Occasional delay
        
        stats = transformer.get_stats()
        
        # Check all expected keys are present
        expected_keys = [
            'fps_current', 'fps_avg', 'fps_std', 'fps_target',
            'latency_current_ms', 'latency_avg_ms', 'latency_std_ms', 'latency_p95_ms',
            'processing_current_ms', 'processing_avg_ms', 'processing_std_ms',
            'frames_total', 'frames_processed', 'frames_dropped', 'drop_rate_percent',
            'consecutive_slow_frames', 'backpressure_threshold_ms'
        ]
        
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
        
        # Check pipeline stats are included
        pipeline_keys = [
            'frames_processed', 'detections_run', 'ocr_requests', 'tracks_active',
            'avg_processing_time', 'ocr_queue_size'
        ]
        
        for key in pipeline_keys:
            assert key in stats
    
    def test_performance_health_check(self, transformer):
        """Test performance health assessment."""
        # Healthy performance
        transformer.fps_ema = 30.0
        transformer.latency_ema = 0.05  # 50ms
        transformer.dropped_frames = 1
        transformer.total_frames = 100
        
        assert transformer.is_performance_healthy() == True
        
        # Unhealthy FPS
        transformer.fps_ema = 15.0  # Below threshold
        assert transformer.is_performance_healthy() == False
        
        # Reset FPS, test high latency
        transformer.fps_ema = 30.0
        transformer.latency_ema = 0.2  # 200ms, above threshold
        assert transformer.is_performance_healthy() == False
        
        # Reset latency, test high drop rate
        transformer.latency_ema = 0.05
        transformer.dropped_frames = 15  # 15% drop rate
        assert transformer.is_performance_healthy() == False
    
    def test_stats_reset(self, transformer, mock_frame):
        """Test statistics reset functionality."""
        # Process some frames first
        with patch('av.VideoFrame.from_ndarray'):
            for i in range(5):
                transformer.recv(mock_frame)
        
        # Verify stats are populated
        stats_before = transformer.get_stats()
        assert stats_before['frames_total'] > 0
        assert stats_before['fps_current'] > 0
        
        # Reset stats
        transformer.reset_stats()
        
        # Verify stats are cleared
        stats_after = transformer.get_stats()
        assert stats_after['frames_total'] == 0
        assert stats_after['frames_processed'] == 0
        assert stats_after['frames_dropped'] == 0
        assert stats_after['fps_current'] == 0.0
        assert stats_after['latency_current_ms'] == 0.0
    
    def test_performance_summary(self, transformer, mock_frame):
        """Test performance summary generation."""
        with patch('av.VideoFrame.from_ndarray'):
            transformer.recv(mock_frame)
        
        summary = transformer.get_performance_summary()
        
        # Should be a formatted string with key metrics
        assert isinstance(summary, str)
        assert 'FPS:' in summary
        assert 'Latency:' in summary
        assert 'Processing:' in summary
        assert 'Frames:' in summary
        assert 'Active Tracks:' in summary
        assert 'OCR Queue:' in summary
    
    def test_consecutive_slow_frames_tracking(self, transformer, mock_frame):
        """Test tracking of consecutive slow frames."""
        # Mock slow processing consistently
        def slow_process(*args):
            time.sleep(0.15)  # Above backpressure threshold
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        
        transformer.pipeline.process_frame.side_effect = slow_process
        
        with patch('av.VideoFrame.from_ndarray'):
            # Process several frames
            for i in range(8):
                transformer.recv(mock_frame)
        
        # Should track consecutive slow frames
        assert transformer.consecutive_slow_frames > 0
        
        # Should start dropping frames after threshold
        stats = transformer.get_stats()
        assert stats['frames_dropped'] > 0
    
    def test_ema_smoothing(self, transformer):
        """Test EMA smoothing of performance metrics."""
        # Simulate frame processing with known timing
        transformer.last_frame_time = time.time()
        
        # First update
        transformer._update_performance_stats(time.time() - 0.1, 0.05)
        first_fps = transformer.fps_ema
        first_latency = transformer.latency_ema
        
        # Second update with different values
        time.sleep(0.01)
        transformer._update_performance_stats(time.time() - 0.08, 0.03)
        
        # EMA should be different from raw values
        assert transformer.fps_ema != first_fps
        assert transformer.latency_ema != first_latency
        
        # Should be smoothed (between old and new values)
        assert transformer.fps_ema > 0
        assert transformer.latency_ema > 0


class TestWebRTCIntegration:
    """Test WebRTC integration scenarios."""
    
    def test_frame_format_conversion(self):
        """Test av.VideoFrame format conversion."""
        # Create test image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Convert to av.VideoFrame
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Convert back to numpy
        converted = frame.to_ndarray(format="bgr24")
        
        # Should be identical
        np.testing.assert_array_equal(img, converted)
    
    def test_frame_timing_preservation(self):
        """Test that frame timing information is preserved."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create frame with timing info
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = 12345
        frame.time_base = av.Rational(1, 30)
        
        # Process through transformer
        config = Config()
        mock_pipeline = Mock(spec=RealtimePipeline)
        mock_pipeline.process_frame.return_value = img
        mock_pipeline.get_stats.return_value = {}
        
        transformer = VideoTransformer(config, mock_pipeline)
        
        with patch('av.VideoFrame.from_ndarray') as mock_from_ndarray:
            mock_output = Mock(spec=av.VideoFrame)
            mock_from_ndarray.return_value = mock_output
            
            result = transformer.recv(frame)
            
            # Timing should be preserved
            assert mock_output.pts == frame.pts
            assert mock_output.time_base == frame.time_base
    
    def test_memory_efficiency(self, transformer, mock_frame):
        """Test memory efficiency with many frames."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch('av.VideoFrame.from_ndarray'):
            # Process many frames
            for i in range(100):
                transformer.recv(mock_frame)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100 * 1024 * 1024
    
    def test_concurrent_processing_safety(self, transformer, mock_frame):
        """Test thread safety of statistics updates."""
        import threading
        
        def process_frames():
            with patch('av.VideoFrame.from_ndarray'):
                for i in range(10):
                    transformer.recv(mock_frame)
        
        # Run multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=process_frames)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should not crash and should have processed frames
        stats = transformer.get_stats()
        assert stats['frames_total'] > 0