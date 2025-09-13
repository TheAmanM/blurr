"""Integration tests for the RealtimePipeline orchestrator."""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

from privacy_redactor_rt.pipeline import RealtimePipeline
from privacy_redactor_rt.config import Config
from privacy_redactor_rt.types import BBox, Detection, Match, Track


class TestRealtimePipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Config()
        # Adjust for faster testing
        config.realtime.detector_stride = 2
        config.realtime.ocr_refresh_stride = 5
        config.classification.require_temporal_consensus = 2
        return config
    
    @pytest.fixture
    def mock_frame(self):
        """Create a mock video frame."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def pipeline(self, config):
        """Create pipeline instance with mocked components."""
        with patch('privacy_redactor_rt.pipeline.TextDetector'), \
             patch('privacy_redactor_rt.pipeline.OCRWorker'), \
             patch('privacy_redactor_rt.pipeline.ClassificationEngine'), \
             patch('privacy_redactor_rt.pipeline.OpticalFlowTracker'), \
             patch('privacy_redactor_rt.pipeline.RedactionEngine'):
            
            pipeline = RealtimePipeline(config)
            
            # Mock the components
            pipeline.text_detector.detect = Mock(return_value=[])
            pipeline.ocr_worker.start = Mock()
            pipeline.ocr_worker.stop = Mock()
            pipeline.ocr_worker.enqueue_roi = Mock(return_value=True)
            pipeline.ocr_worker.get_result = Mock(return_value=None)
            pipeline.classifier.classify_text = Mock(return_value=[])
            pipeline.tracker.propagate_tracks = Mock()
            pipeline.tracker.associate_detections = Mock()
            pipeline.tracker.get_all_tracks = Mock(return_value=[])
            pipeline.tracker.get_active_tracks = Mock(return_value=[])
            pipeline.tracker.cleanup_tracks = Mock()
            pipeline.tracker.reset = Mock()
            pipeline.tracker.should_ocr_track = Mock(return_value=False)
            pipeline.tracker.update_track_ocr_frame = Mock()
            pipeline.redactor.redact_regions = Mock(side_effect=lambda frame, tracks: frame)
            
            return pipeline
    
    def test_pipeline_initialization(self, config):
        """Test pipeline initializes all components correctly."""
        with patch('privacy_redactor_rt.pipeline.TextDetector') as mock_detector, \
             patch('privacy_redactor_rt.pipeline.OCRWorker') as mock_ocr, \
             patch('privacy_redactor_rt.pipeline.ClassificationEngine') as mock_classifier, \
             patch('privacy_redactor_rt.pipeline.OpticalFlowTracker') as mock_tracker, \
             patch('privacy_redactor_rt.pipeline.RedactionEngine') as mock_redactor:
            
            pipeline = RealtimePipeline(config)
            
            # Verify all components were initialized
            mock_detector.assert_called_once_with(config.detection)
            mock_ocr.assert_called_once_with(config.ocr, config.realtime.max_queue)
            mock_classifier.assert_called_once_with(config.classification)
            mock_tracker.assert_called_once_with(config.tracking)
            mock_redactor.assert_called_once_with(config.redaction)
            
            assert pipeline.frame_count == 0
            assert pipeline.prev_frame is None
            assert pipeline.consensus_buffer == {}
    
    def test_start_stop_lifecycle(self, pipeline):
        """Test pipeline start/stop lifecycle."""
        # Test start
        pipeline.start()
        pipeline.ocr_worker.start.assert_called_once()
        
        # Test stop
        pipeline.stop()
        pipeline.ocr_worker.stop.assert_called_once()
    
    def test_context_manager(self, pipeline):
        """Test pipeline as context manager."""
        with pipeline:
            pipeline.ocr_worker.start.assert_called_once()
        pipeline.ocr_worker.stop.assert_called_once()
    
    def test_detection_scheduling(self, pipeline):
        """Test detection scheduling based on frame stride."""
        # Frame 0: should run detection (0 % 2 == 0)
        assert pipeline.should_run_detection(0) is True
        
        # Frame 1: should not run detection (1 % 2 != 0)
        assert pipeline.should_run_detection(1) is False
        
        # Frame 2: should run detection (2 % 2 == 0)
        assert pipeline.should_run_detection(2) is True
    
    def test_process_frame_basic_flow(self, pipeline, mock_frame):
        """Test basic frame processing flow."""
        # Setup mocks
        mock_bboxes = [BBox(10, 10, 50, 30, 0.8)]
        pipeline.text_detector.detect.return_value = mock_bboxes
        
        # Process frame
        result = pipeline.process_frame(mock_frame, 0)
        
        # Verify basic flow
        assert result is not None
        assert pipeline.frame_count == 0
        assert pipeline.prev_frame is not None
        
        # Verify detection was called (frame 0 should trigger detection)
        pipeline.text_detector.detect.assert_called_once_with(mock_frame)
        
        # Verify tracking was called
        pipeline.tracker.associate_detections.assert_called_once()
        
        # Verify redaction was called
        pipeline.redactor.redact_regions.assert_called_once()
    
    def test_process_frame_with_optical_flow(self, pipeline, mock_frame):
        """Test frame processing with optical flow propagation."""
        # Process first frame
        pipeline.process_frame(mock_frame, 0)
        
        # Process second frame (should trigger optical flow)
        mock_frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pipeline.process_frame(mock_frame2, 1)
        
        # Verify optical flow was called
        pipeline.tracker.propagate_tracks.assert_called_with(mock_frame2, mock_frame)
    
    def test_ocr_scheduling(self, pipeline, mock_frame):
        """Test OCR scheduling for tracks."""
        # Create mock track that needs OCR
        mock_track = Track(
            id="test_track",
            bbox=BBox(10, 10, 50, 30, 0.8),
            matches=[],
            age=1,
            hits=3,
            last_ocr_frame=-1
        )
        
        pipeline.tracker.get_all_tracks.return_value = [mock_track]
        pipeline.tracker.should_ocr_track.return_value = True
        
        # Process frame
        pipeline.process_frame(mock_frame, 0)
        
        # Verify OCR was scheduled
        pipeline.ocr_worker.enqueue_roi.assert_called_once()
        pipeline.tracker.update_track_ocr_frame.assert_called_once_with("test_track", 0)
    
    def test_classification_and_consensus(self, pipeline, mock_frame):
        """Test classification and temporal consensus logic."""
        # Create mock track with OCR result
        mock_track = Track(
            id="test_track",
            bbox=BBox(10, 10, 50, 30, 0.8),
            matches=[],
            age=1,
            hits=3,
            last_ocr_frame=0
        )
        
        # Mock OCR result
        pipeline.ocr_worker.get_result.return_value = "test@example.com"
        
        # Mock classification result
        mock_match = Match(
            category="email",
            confidence=0.9,
            masked_text="t***@example.com",
            bbox=BBox(10, 10, 50, 30, 0.8)
        )
        pipeline.classifier.classify_text.return_value = [mock_match]
        
        pipeline.tracker.get_all_tracks.return_value = [mock_track]
        pipeline.tracker.get_active_tracks.return_value = [mock_track]
        
        # Process frame to build consensus
        pipeline.process_frame(mock_frame, 0)
        
        # Verify classification was called
        pipeline.classifier.classify_text.assert_called_with("test@example.com", mock_track.bbox)
        
        # Check consensus buffer
        assert "test_track" in pipeline.consensus_buffer
        assert len(pipeline.consensus_buffer["test_track"]) == 1
    
    def test_temporal_consensus_filtering(self, pipeline, mock_frame):
        """Test temporal consensus filtering."""
        # Create track with insufficient consensus
        track_id = "test_track"
        mock_track = Track(
            id=track_id,
            bbox=BBox(10, 10, 50, 30, 0.8),
            matches=[],
            age=1,
            hits=3,
            last_ocr_frame=0
        )
        
        # Add only one match (need 2 for consensus)
        mock_match = Match(
            category="email",
            confidence=0.9,
            masked_text="t***@example.com",
            bbox=BBox(10, 10, 50, 30, 0.8)
        )
        pipeline.consensus_buffer[track_id] = [mock_match]
        
        # Should not have consensus
        assert not pipeline._has_temporal_consensus(mock_track)
        
        # Add second match
        pipeline.consensus_buffer[track_id].append(mock_match)
        
        # Should now have consensus
        assert pipeline._has_temporal_consensus(mock_track)
    
    def test_error_handling(self, pipeline, mock_frame):
        """Test error handling in frame processing."""
        # Make text detector raise an exception
        pipeline.text_detector.detect.side_effect = Exception("Detection failed")
        
        # Process frame should not crash
        result = pipeline.process_frame(mock_frame, 0)
        
        # Should return original frame on error
        assert np.array_equal(result, mock_frame)
    
    def test_statistics_tracking(self, pipeline, mock_frame):
        """Test performance statistics tracking."""
        # Process a few frames
        for i in range(3):
            pipeline.process_frame(mock_frame, i)
        
        stats = pipeline.get_stats()
        
        # Verify basic stats
        assert stats['frames_processed'] == 3
        assert stats['avg_processing_time'] > 0
        assert 'ocr_queue_size' in stats
        assert 'total_tracks' in stats
    
    def test_reset_functionality(self, pipeline):
        """Test pipeline reset functionality."""
        # Set some state
        pipeline.frame_count = 10
        pipeline.prev_frame = np.zeros((100, 100, 3))
        pipeline.consensus_buffer["test"] = []
        pipeline.stats['frames_processed'] = 5
        
        # Reset
        pipeline.reset()
        
        # Verify state is reset
        assert pipeline.frame_count == 0
        assert pipeline.prev_frame is None
        assert pipeline.consensus_buffer == {}
        assert pipeline.stats['frames_processed'] == 0
        
        # Verify tracker reset was called
        pipeline.tracker.reset.assert_called_once()
    
    def test_consensus_buffer_cleanup(self, pipeline, mock_frame):
        """Test consensus buffer cleanup for expired tracks."""
        # Add entry for non-existent track
        pipeline.consensus_buffer["expired_track"] = [Mock()]
        
        # Process frame (should trigger cleanup)
        pipeline.process_frame(mock_frame, 0)
        
        # Expired track should be removed
        assert "expired_track" not in pipeline.consensus_buffer
    
    def test_roi_extraction(self, pipeline):
        """Test ROI extraction from frames."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = BBox(10, 10, 50, 50, 0.8)
        
        roi = pipeline._extract_roi(frame, bbox)
        
        assert roi is not None
        assert roi.shape == (40, 40, 3)  # 50-10 = 40 for both dimensions
    
    def test_roi_extraction_boundary_conditions(self, pipeline):
        """Test ROI extraction with boundary conditions."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test bbox outside frame bounds
        bbox = BBox(150, 150, 200, 200, 0.8)
        roi = pipeline._extract_roi(frame, bbox)
        assert roi is None  # Should be None due to clamping resulting in invalid size
        
        # Test very small bbox
        bbox = BBox(10, 10, 11, 11, 0.8)
        roi = pipeline._extract_roi(frame, bbox)
        assert roi is not None
        assert roi.shape == (1, 1, 3)


class TestPipelineEndToEnd:
    """End-to-end integration tests with real components."""
    
    @pytest.fixture
    def real_config(self):
        """Create configuration for real component testing."""
        config = Config()
        config.realtime.detector_stride = 1  # Run detection every frame for testing
        config.classification.require_temporal_consensus = 1  # Minimal consensus for testing
        return config
    
    @pytest.mark.integration
    def test_end_to_end_with_mocked_dependencies(self, real_config):
        """Test end-to-end pipeline with mocked external dependencies."""
        # Mock external dependencies that require installation
        with patch('privacy_redactor_rt.text_detect.PaddleOCR'), \
             patch('privacy_redactor_rt.ocr.PaddleOCR'), \
             patch('privacy_redactor_rt.track.cv2') as mock_cv2:
            
            # Setup cv2 mocks
            mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_cv2.calcOpticalFlowPyrLK.return_value = (
                np.array([[10, 10]], dtype=np.float32),  # new points
                np.array([[1]], dtype=np.uint8),         # status
                np.array([[1.0]], dtype=np.float32)      # error
            )
            mock_cv2.goodFeaturesToTrack.return_value = np.array([[[5, 5]]], dtype=np.float32)
            mock_cv2.GaussianBlur.side_effect = lambda img, ksize, sigmaX, sigmaY: img
            
            # Create pipeline
            pipeline = RealtimePipeline(real_config)
            
            # Create test frame
            test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            try:
                pipeline.start()
                
                # Process several frames
                for i in range(5):
                    result = pipeline.process_frame(test_frame, i)
                    assert result is not None
                    assert result.shape == test_frame.shape
                
                # Verify stats are being tracked
                stats = pipeline.get_stats()
                assert stats['frames_processed'] == 5
                
            finally:
                pipeline.stop()
    
    def test_pipeline_performance_under_load(self, real_config):
        """Test pipeline performance with multiple frames."""
        with patch('privacy_redactor_rt.text_detect.PaddleOCR'), \
             patch('privacy_redactor_rt.ocr.PaddleOCR'), \
             patch('privacy_redactor_rt.track.cv2') as mock_cv2:
            
            # Setup minimal cv2 mocks
            mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_cv2.GaussianBlur.side_effect = lambda img, ksize, sigmaX, sigmaY: img
            
            pipeline = RealtimePipeline(real_config)
            test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            try:
                pipeline.start()
                
                # Process many frames and measure performance
                start_time = time.time()
                num_frames = 20
                
                for i in range(num_frames):
                    pipeline.process_frame(test_frame, i)
                
                total_time = time.time() - start_time
                fps = num_frames / total_time
                
                # Should be able to process at reasonable speed
                assert fps > 10  # At least 10 FPS with mocked components
                
                stats = pipeline.get_stats()
                assert stats['frames_processed'] == num_frames
                assert stats['avg_processing_time'] > 0
                
            finally:
                pipeline.stop()