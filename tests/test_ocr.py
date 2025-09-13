"""Unit tests for OCR worker with queue management and text processing."""

import pytest
import threading
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from privacy_redactor_rt.ocr import OCRWorker
from privacy_redactor_rt.config import OCRConfig
from privacy_redactor_rt.types import BBox


class TestOCRWorker:
    """Test cases for OCRWorker class."""
    
    @pytest.fixture
    def ocr_config(self):
        """Create OCR configuration for testing."""
        return OCRConfig(
            min_ocr_confidence=0.7,
            use_gpu=False,
            rec_batch_num=1,
            max_text_length=100,
            enable_mkldnn=False,
            cpu_threads=1
        )
    
    @pytest.fixture
    def sample_roi(self):
        """Create sample ROI image for testing."""
        # Create a simple 3-channel BGR image
        return np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_bbox(self):
        """Create sample bounding box for testing."""
        return BBox(x1=10, y1=20, x2=110, y2=70, confidence=0.8)
    
    def test_initialization(self, ocr_config):
        """Test OCR worker initialization."""
        worker = OCRWorker(ocr_config, max_queue_size=5)
        
        assert worker.config == ocr_config
        assert worker.max_queue_size == 5
        assert worker._worker_thread is None
        assert not worker._stop_event.is_set()
        assert worker._queue.maxsize == 5
        assert len(worker._results) == 0
    
    def test_start_stop_worker(self, ocr_config):
        """Test starting and stopping the worker thread."""
        worker = OCRWorker(ocr_config)
        
        # Start worker
        worker.start()
        assert worker._worker_thread is not None
        assert worker._worker_thread.is_alive()
        
        # Stop worker
        worker.stop()
        assert not worker._worker_thread.is_alive()
    
    def test_context_manager(self, ocr_config):
        """Test OCR worker as context manager."""
        with OCRWorker(ocr_config) as worker:
            assert worker._worker_thread is not None
            assert worker._worker_thread.is_alive()
        
        # Should be stopped after exiting context
        assert not worker._worker_thread.is_alive()
    
    def test_enqueue_roi_success(self, ocr_config, sample_roi, sample_bbox):
        """Test successful ROI enqueuing."""
        worker = OCRWorker(ocr_config, max_queue_size=5)
        
        result = worker.enqueue_roi(sample_roi, "track_1", sample_bbox)
        assert result is True
        assert worker._queue.qsize() == 1
    
    def test_enqueue_roi_invalid_input(self, ocr_config, sample_bbox):
        """Test enqueuing invalid ROI."""
        worker = OCRWorker(ocr_config)
        
        # Test None ROI
        result = worker.enqueue_roi(None, "track_1", sample_bbox)
        assert result is False
        
        # Test empty ROI
        empty_roi = np.array([])
        result = worker.enqueue_roi(empty_roi, "track_1", sample_bbox)
        assert result is False
        
        # Test invalid shape (2D instead of 3D)
        invalid_roi = np.random.randint(0, 255, (50, 100), dtype=np.uint8)
        result = worker.enqueue_roi(invalid_roi, "track_1", sample_bbox)
        assert result is False
    
    def test_queue_full_handling(self, ocr_config, sample_roi, sample_bbox):
        """Test queue full handling with backpressure."""
        worker = OCRWorker(ocr_config, max_queue_size=2)
        
        # Fill the queue
        assert worker.enqueue_roi(sample_roi, "track_1", sample_bbox) is True
        assert worker.enqueue_roi(sample_roi, "track_2", sample_bbox) is True
        
        # Next enqueue should fail (queue full)
        assert worker.enqueue_roi(sample_roi, "track_3", sample_bbox) is False
        
        # Check statistics
        stats = worker.get_stats()
        assert stats['queue_full_count'] == 1
    
    def test_result_caching(self, ocr_config):
        """Test result caching and retrieval."""
        worker = OCRWorker(ocr_config)
        
        # Initially no result
        assert worker.get_result("track_1") is None
        
        # Manually add result to cache
        with worker._results_lock:
            worker._results["track_1"] = ("test text", time.time())
        
        # Should retrieve cached result
        result = worker.get_result("track_1")
        assert result == "test text"
        
        # Check cache hit statistics
        stats = worker.get_stats()
        assert stats['cache_hits'] == 1
    
    def test_clear_result(self, ocr_config):
        """Test clearing cached results."""
        worker = OCRWorker(ocr_config)
        
        # Add result to cache
        with worker._results_lock:
            worker._results["track_1"] = ("test text", time.time())
        
        # Clear result
        worker.clear_result("track_1")
        
        # Should be gone
        assert worker.get_result("track_1") is None
    
    def test_statistics(self, ocr_config):
        """Test statistics collection."""
        worker = OCRWorker(ocr_config, max_queue_size=3)
        
        stats = worker.get_stats()
        expected_keys = ['processed_count', 'queue_full_count', 'cache_hits', 'queue_size', 'cache_size']
        
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], int)
    
    @patch('privacy_redactor_rt.ocr.PaddleOCR')
    def test_lazy_ocr_initialization(self, mock_paddle_ocr, ocr_config):
        """Test lazy initialization of OCR engine."""
        mock_ocr_instance = Mock()
        mock_paddle_ocr.return_value = mock_ocr_instance
        
        worker = OCRWorker(ocr_config)
        
        # OCR should not be initialized yet
        assert worker._ocr_engine is None
        
        # Initialize OCR
        ocr_engine = worker._lazy_init_ocr()
        
        # Should be initialized now
        assert worker._ocr_engine is not None
        assert ocr_engine == mock_ocr_instance
        
        # Verify PaddleOCR was called with correct parameters
        mock_paddle_ocr.assert_called_once_with(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            rec_batch_num=1,
            enable_mkldnn=False,
            cpu_threads=1,
            show_log=False
        )
    
    def test_text_normalization(self, ocr_config):
        """Test text normalization functionality."""
        worker = OCRWorker(ocr_config)
        
        # Test basic normalization
        text = "  Hello   World  \n\t "
        normalized = worker._normalize_text(text)
        assert normalized == "Hello World"
        
        # Test Unicode normalization (NFKC)
        text = "café"  # Contains combining characters
        normalized = worker._normalize_text(text)
        assert normalized == "café"
        
        # Test empty string
        assert worker._normalize_text("") == ""
        assert worker._normalize_text(None) == ""
    
    @patch('privacy_redactor_rt.ocr.PaddleOCR')
    def test_extract_and_normalize_text(self, mock_paddle_ocr, ocr_config):
        """Test text extraction and normalization from OCR results."""
        worker = OCRWorker(ocr_config)
        
        # Mock OCR results format
        ocr_results = [[
            [[[0, 0], [100, 0], [100, 30], [0, 30]], ("Hello World", 0.95)],
            [[[0, 35], [80, 35], [80, 65], [0, 65]], ("Test Text", 0.85)],
            [[[0, 70], [60, 70], [60, 100], [0, 100]], ("Low Conf", 0.5)]  # Below threshold
        ]]
        
        text = worker._extract_and_normalize_text(ocr_results)
        
        # Should include high confidence text, exclude low confidence
        assert "Hello World" in text
        assert "Test Text" in text
        assert "Low Conf" not in text
    
    def test_extract_text_empty_results(self, ocr_config):
        """Test text extraction with empty OCR results."""
        worker = OCRWorker(ocr_config)
        
        # Test various empty result formats
        assert worker._extract_and_normalize_text(None) == ""
        assert worker._extract_and_normalize_text([]) == ""
        assert worker._extract_and_normalize_text([[]]) == ""
    
    @patch('privacy_redactor_rt.ocr.PaddleOCR')
    def test_process_ocr_request(self, mock_paddle_ocr, ocr_config, sample_roi, sample_bbox):
        """Test processing of OCR requests."""
        mock_ocr_instance = Mock()
        mock_ocr_results = [[
            [[[0, 0], [100, 0], [100, 30], [0, 30]], ("Test Result", 0.9)]
        ]]
        mock_ocr_instance.ocr.return_value = mock_ocr_results
        mock_paddle_ocr.return_value = mock_ocr_instance
        
        worker = OCRWorker(ocr_config)
        
        # Create request item
        item = {
            'roi': sample_roi,
            'track_id': 'test_track',
            'bbox': sample_bbox,
            'timestamp': time.time()
        }
        
        # Process request
        worker._process_ocr_request(item)
        
        # Check that result was cached
        result = worker.get_result('test_track')
        assert result == "Test Result"
        
        # Check statistics
        stats = worker.get_stats()
        assert stats['processed_count'] == 1
    
    def test_cleanup_expired_results(self, ocr_config):
        """Test cleanup of expired cached results."""
        worker = OCRWorker(ocr_config)
        
        current_time = time.time()
        
        # Add some results with different timestamps
        with worker._results_lock:
            worker._results["fresh"] = ("fresh text", current_time)
            worker._results["old"] = ("old text", current_time - 400)  # Older than 300s
            worker._results["ancient"] = ("ancient text", current_time - 600)
        
        # Cleanup with 300s max age
        cleaned_count = worker.cleanup_expired_results(max_age_seconds=300.0)
        
        assert cleaned_count == 2  # Should remove "old" and "ancient"
        assert worker.get_result("fresh") == "fresh text"
        assert worker.get_result("old") is None
        assert worker.get_result("ancient") is None
    
    @patch('privacy_redactor_rt.ocr.PaddleOCR')
    def test_integration_with_worker_thread(self, mock_paddle_ocr, ocr_config, sample_roi, sample_bbox):
        """Test integration with actual worker thread processing."""
        mock_ocr_instance = Mock()
        mock_ocr_results = [[
            [[[0, 0], [100, 0], [100, 30], [0, 30]], ("Integration Test", 0.9)]
        ]]
        mock_ocr_instance.ocr.return_value = mock_ocr_results
        mock_paddle_ocr.return_value = mock_ocr_instance
        
        with OCRWorker(ocr_config, max_queue_size=5) as worker:
            # Enqueue ROI for processing
            success = worker.enqueue_roi(sample_roi, "integration_track", sample_bbox)
            assert success is True
            
            # Wait for processing (with timeout)
            max_wait = 5.0
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                result = worker.get_result("integration_track")
                if result is not None:
                    break
                time.sleep(0.1)
            
            # Should have processed the ROI
            result = worker.get_result("integration_track")
            assert result == "Integration Test"
            
            # Check final statistics
            stats = worker.get_stats()
            assert stats['processed_count'] >= 1
    
    def test_thread_safety(self, ocr_config, sample_roi, sample_bbox):
        """Test thread safety of OCR worker operations."""
        worker = OCRWorker(ocr_config, max_queue_size=10)
        worker.start()
        
        results = []
        errors = []
        
        def enqueue_worker(worker_id):
            """Worker function for threading test."""
            try:
                for i in range(5):
                    track_id = f"thread_{worker_id}_track_{i}"
                    success = worker.enqueue_roi(sample_roi, track_id, sample_bbox)
                    results.append((worker_id, i, success))
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=enqueue_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        worker.stop()
        
        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 15  # 3 threads * 5 operations each
        
        # All operations should have succeeded or failed gracefully
        for worker_id, op_id, success in results:
            assert isinstance(success, bool)


class TestOCRTextNormalization:
    """Specific tests for text normalization functionality."""
    
    @pytest.fixture
    def ocr_config(self):
        """Create OCR configuration for testing."""
        return OCRConfig()
    
    def test_unicode_normalization(self, ocr_config):
        """Test Unicode NFKC normalization."""
        worker = OCRWorker(ocr_config)
        
        # Test combining characters
        text = "e\u0301"  # e + combining acute accent
        normalized = worker._normalize_text(text)
        assert normalized == "é"
        
        # Test compatibility characters
        text = "ﬁ"  # fi ligature (U+FB01)
        normalized = worker._normalize_text(text)
        assert normalized == "fi"
    
    def test_whitespace_normalization(self, ocr_config):
        """Test whitespace handling."""
        worker = OCRWorker(ocr_config)
        
        test_cases = [
            ("  hello  world  ", "hello world"),
            ("hello\n\nworld", "hello world"),
            ("hello\t\tworld", "hello world"),
            ("hello\r\nworld", "hello world"),
            ("   ", ""),
            ("\n\t\r", ""),
        ]
        
        for input_text, expected in test_cases:
            result = worker._normalize_text(input_text)
            assert result == expected, f"Failed for input: '{input_text}'"
    
    def test_case_preservation(self, ocr_config):
        """Test that case is preserved during normalization."""
        worker = OCRWorker(ocr_config)
        
        text = "Hello WORLD Test"
        normalized = worker._normalize_text(text)
        assert normalized == "Hello WORLD Test"
    
    def test_text_length_truncation(self, ocr_config):
        """Test text length truncation."""
        config = OCRConfig(max_text_length=10)
        worker = OCRWorker(config)
        
        # Mock OCR results with long text
        ocr_results = [[
            [[[0, 0], [100, 0], [100, 30], [0, 30]], ("This is a very long text that should be truncated", 0.9)]
        ]]
        
        text = worker._extract_and_normalize_text(ocr_results)
        assert len(text) <= 10
        assert text == "This is a "  # Truncated at 10 characters


if __name__ == "__main__":
    pytest.main([__file__])