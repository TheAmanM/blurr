"""Integration tests for OCR worker with mocked PaddleOCR."""

import time
import numpy as np
from unittest.mock import Mock, patch
import pytest

from privacy_redactor_rt.ocr import OCRWorker
from privacy_redactor_rt.config import OCRConfig
from privacy_redactor_rt.types import BBox


@patch('privacy_redactor_rt.ocr.PADDLEOCR_AVAILABLE', True)
@patch('privacy_redactor_rt.ocr.PaddleOCR')
def test_ocr_integration_with_mocked_paddle(mock_paddle_ocr):
    """Test OCR worker integration with mocked PaddleOCR."""
    # Setup mock
    mock_ocr_instance = Mock()
    mock_ocr_results = [[
        [[[0, 0], [100, 0], [100, 30], [0, 30]], ("Test Result", 0.9)]
    ]]
    mock_ocr_instance.ocr.return_value = mock_ocr_results
    mock_paddle_ocr.return_value = mock_ocr_instance
    
    config = OCRConfig(min_ocr_confidence=0.7)
    
    with OCRWorker(config, max_queue_size=5) as worker:
        # Create sample ROI and bbox
        roi = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
        bbox = BBox(x1=10, y1=20, x2=110, y2=70, confidence=0.8)
        
        # Enqueue ROI for processing
        success = worker.enqueue_roi(roi, "integration_track", bbox)
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
        assert result == "Test Result"
        
        # Verify PaddleOCR was initialized correctly
        mock_paddle_ocr.assert_called_once_with(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            rec_batch_num=6,  # Default from OCRConfig
            enable_mkldnn=True,
            cpu_threads=4,
            show_log=False
        )
        
        # Verify OCR was called
        mock_ocr_instance.ocr.assert_called_once()


@patch('privacy_redactor_rt.ocr.PADDLEOCR_AVAILABLE', False)
def test_ocr_worker_without_paddleocr():
    """Test OCR worker behavior when PaddleOCR is not available."""
    config = OCRConfig()
    worker = OCRWorker(config)
    
    # Should raise ImportError when trying to initialize OCR
    with pytest.raises(ImportError, match="PaddleOCR is not available"):
        worker._lazy_init_ocr()


if __name__ == "__main__":
    # Run the integration test
    test_ocr_integration_with_mocked_paddle()
    test_ocr_worker_without_paddleocr()
    print("✅ OCR integration tests passed!")