"""Integration tests for complete application workflow."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import threading
import time

from privacy_redactor_rt.config import Config

# Mock streamlit and webrtc dependencies for testing
with patch.dict('sys.modules', {
    'streamlit': Mock(),
    'streamlit_webrtc': Mock(),
    'av': Mock()
}):
    from privacy_redactor_rt.app import StreamlitApp
    from privacy_redactor_rt.pipeline import RealtimePipeline
    from privacy_redactor_rt.webrtc_utils import VideoTransformer


class TestAppIntegration:
    """Test complete application integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
        # Create test configuration
        self.config = Config()
        self.config.to_yaml(self.config_path)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('privacy_redactor_rt.app.st')
    def test_app_initialization(self, mock_st):
        """Test application initialization."""
        # Mock Streamlit session state
        mock_st.session_state = {}
        
        # Create app instance
        app = StreamlitApp()
        
        # Verify initialization
        assert app.config is None
        assert app.pipeline is None
        assert app.video_transformer is None
        assert app.detection_counters == {}
        assert app.event_feed == []
        assert app.max_events == 50
        assert not app._initialized
        
        # Verify session state initialization
        assert 'config_loaded' in mock_st.session_state
        assert 'pipeline_active' in mock_st.session_state
        assert 'recording_active' in mock_st.session_state
        assert 'app_instance' in mock_st.session_state
    
    @patch('privacy_redactor_rt.app.st')
    @patch('privacy_redactor_rt.app.load_config')
    def test_configuration_loading(self, mock_load_config, mock_st):
        """Test configuration loading with various scenarios."""
        mock_st.session_state = {}
        mock_st.error = Mock()
        mock_st.warning = Mock()
        
        app = StreamlitApp()
        
        # Test successful loading
        mock_load_config.return_value = self.config
        app._load_configuration()
        
        assert app.config is not None
        assert app._initialized
        mock_load_config.assert_called_once()
        
        # Test file not found (should use default config)
        mock_load_config.side_effect = FileNotFoundError("Config not found")
        app._load_configuration()
        
        # Should still have a config (default)
        assert app.config is not None
        mock_st.error.assert_called()
    
    @patch('privacy_redactor_rt.app.st')
    def test_pipeline_initialization(self, mock_st):
        """Test pipeline initialization and error handling."""
        mock_st.session_state = {'pipeline_active': False}
        mock_st.error = Mock()
        mock_st.spinner = Mock()
        
        app = StreamlitApp()
        app.config = self.config
        
        # Test successful initialization
        with patch('privacy_redactor_rt.app.RealtimePipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            
            app._initialize_pipeline()
            
            assert app.pipeline is not None
            mock_pipeline.start.assert_called_once()
            assert mock_st.session_state['pipeline_active']
        
        # Test initialization failure
        with patch('privacy_redactor_rt.app.RealtimePipeline') as mock_pipeline_class:
            mock_pipeline_class.side_effect = Exception("Pipeline init failed")
            
            app._initialize_pipeline()
            
            assert app.pipeline is None
            mock_st.error.assert_called()
    
    @patch('privacy_redactor_rt.app.st')
    def test_video_transformer_creation(self, mock_st):
        """Test video transformer creation and integration."""
        mock_st.session_state = {}
        
        app = StreamlitApp()
        app.config = self.config
        
        # Mock pipeline
        mock_pipeline = Mock()
        app.pipeline = mock_pipeline
        
        # Test transformer creation
        with patch('privacy_redactor_rt.app.VideoTransformer') as mock_transformer_class:
            mock_transformer = Mock()
            mock_transformer_class.return_value = mock_transformer
            
            # Simulate transformer creation in video interface
            app.video_transformer = VideoTransformer(
                app.config, 
                app.pipeline, 
                event_callback=app.add_detection_event
            )
            
            assert app.video_transformer is not None
    
    def test_detection_event_handling(self):
        """Test detection event handling with thread safety."""
        app = StreamlitApp()
        
        # Test single event
        app.add_detection_event("phone", "555-***-1234", 0.95)
        
        assert len(app.event_feed) == 1
        assert app.detection_counters["phone"] == 1
        
        event = app.event_feed[0]
        assert event["category"] == "phone"
        assert event["masked_text"] == "555-***-1234"
        assert event["confidence"] == 0.95
        assert "timestamp" in event
        
        # Test multiple events
        app.add_detection_event("email", "test@***.com", 0.88)
        app.add_detection_event("phone", "555-***-5678", 0.92)
        
        assert len(app.event_feed) == 3
        assert app.detection_counters["phone"] == 2
        assert app.detection_counters["email"] == 1
        
        # Test event limit
        for i in range(60):  # Exceed max_events (50)
            app.add_detection_event("test", f"test-{i}", 0.5)
        
        assert len(app.event_feed) == app.max_events
    
    def test_thread_safety(self):
        """Test thread safety of detection event handling."""
        app = StreamlitApp()
        
        def add_events(category, count):
            for i in range(count):
                app.add_detection_event(category, f"text-{i}", 0.5)
                time.sleep(0.001)  # Small delay to encourage race conditions
        
        # Create multiple threads adding events
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_events, args=(f"category_{i}", 10))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all events were added correctly
        total_events = sum(app.detection_counters.values())
        assert total_events == 50  # 5 categories * 10 events each
        assert len(app.event_feed) == 50
    
    @patch('privacy_redactor_rt.app.st')
    def test_configuration_ui_overrides(self, mock_st):
        """Test UI configuration overrides."""
        # Mock session state with UI values
        mock_st.session_state = {
            'selected_categories': ['phone', 'email'],
            'default_redaction_method': 'pixelate',
            'detector_stride': 5,
            'text_confidence': 0.8,
            'recording_enabled': True,
            'recording_crf': 20
        }
        
        app = StreamlitApp()
        overrides = app._get_ui_config_overrides()
        
        # Verify overrides structure
        assert 'classification' in overrides
        assert 'redaction' in overrides
        assert 'realtime' in overrides
        assert 'detection' in overrides
        assert 'recording' in overrides
        
        # Verify specific values
        assert overrides['classification']['categories'] == ['phone', 'email']
        assert overrides['redaction']['default_method'] == 'pixelate'
        assert overrides['realtime']['detector_stride'] == 5
        assert overrides['detection']['min_text_confidence'] == 0.8
        assert overrides['recording']['enabled'] is True
        assert overrides['recording']['crf'] == 20
    
    def test_cleanup_procedures(self):
        """Test application cleanup procedures."""
        app = StreamlitApp()
        
        # Mock pipeline and transformer
        mock_pipeline = Mock()
        mock_transformer = Mock()
        
        app.pipeline = mock_pipeline
        app.video_transformer = mock_transformer
        
        # Test cleanup
        app._cleanup_on_exit()
        
        # Verify cleanup calls
        mock_pipeline.stop.assert_called_once()
        mock_transformer.reset_stats.assert_called_once()
    
    def test_error_handling_in_monitoring(self):
        """Test error handling in monitoring panel."""
        app = StreamlitApp()
        
        # Mock transformer that raises exceptions
        mock_transformer = Mock()
        mock_transformer.get_stats.side_effect = Exception("Stats error")
        mock_transformer.is_performance_healthy.side_effect = Exception("Health check error")
        
        app.video_transformer = mock_transformer
        app.config = self.config
        
        # Mock Streamlit functions
        with patch('privacy_redactor_rt.app.st') as mock_st:
            mock_st.subheader = Mock()
            mock_st.columns = Mock(return_value=[Mock(), Mock()])
            mock_st.metric = Mock()
            mock_st.write = Mock()
            mock_st.success = Mock()
            mock_st.warning = Mock()
            mock_st.error = Mock()
            mock_st.info = Mock()
            mock_st.divider = Mock()
            mock_st.container = Mock()
            
            # Should not raise exception
            app._create_monitoring_panel()
            
            # Should have called error display
            mock_st.error.assert_called()


class TestPipelineIntegration:
    """Test pipeline integration with all components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        # Use minimal configuration for testing
        self.config.realtime.detector_stride = 1
        self.config.realtime.ocr_refresh_stride = 1
        self.config.classification.require_temporal_consensus = 1
    
    @patch('privacy_redactor_rt.pipeline.TextDetector')
    @patch('privacy_redactor_rt.pipeline.OCRWorker')
    @patch('privacy_redactor_rt.pipeline.ClassificationEngine')
    @patch('privacy_redactor_rt.pipeline.OpticalFlowTracker')
    @patch('privacy_redactor_rt.pipeline.RedactionEngine')
    def test_pipeline_component_integration(self, mock_redactor, mock_tracker, 
                                          mock_classifier, mock_ocr, mock_detector):
        """Test integration of all pipeline components."""
        # Set up mocks
        mock_detector_instance = Mock()
        mock_ocr_instance = Mock()
        mock_classifier_instance = Mock()
        mock_tracker_instance = Mock()
        mock_redactor_instance = Mock()
        
        mock_detector.return_value = mock_detector_instance
        mock_ocr.return_value = mock_ocr_instance
        mock_classifier.return_value = mock_classifier_instance
        mock_tracker.return_value = mock_tracker_instance
        mock_redactor.return_value = mock_redactor_instance
        
        # Create pipeline
        pipeline = RealtimePipeline(self.config)
        
        # Verify all components were created
        mock_detector.assert_called_once_with(self.config.detection)
        mock_ocr.assert_called_once_with(self.config.ocr, self.config.realtime.max_queue)
        mock_classifier.assert_called_once_with(self.config.classification)
        mock_tracker.assert_called_once_with(self.config.tracking)
        mock_redactor.assert_called_once_with(self.config.redaction)
        
        # Test pipeline start/stop
        pipeline.start()
        mock_ocr_instance.start.assert_called_once()
        
        pipeline.stop()
        mock_ocr_instance.stop.assert_called_once()
    
    @patch('privacy_redactor_rt.pipeline.TextDetector')
    @patch('privacy_redactor_rt.pipeline.OCRWorker')
    @patch('privacy_redactor_rt.pipeline.ClassificationEngine')
    @patch('privacy_redactor_rt.pipeline.OpticalFlowTracker')
    @patch('privacy_redactor_rt.pipeline.RedactionEngine')
    def test_frame_processing_workflow(self, mock_redactor, mock_tracker, 
                                     mock_classifier, mock_ocr, mock_detector):
        """Test complete frame processing workflow."""
        # Set up mocks with realistic behavior
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = []  # No detections
        
        mock_ocr_instance = Mock()
        mock_ocr_instance.get_result.return_value = None
        
        mock_classifier_instance = Mock()
        mock_classifier_instance.classify_text.return_value = []
        
        mock_tracker_instance = Mock()
        mock_tracker_instance.get_all_tracks.return_value = []
        mock_tracker_instance.get_active_tracks.return_value = []
        
        mock_redactor_instance = Mock()
        
        mock_detector.return_value = mock_detector_instance
        mock_ocr.return_value = mock_ocr_instance
        mock_classifier.return_value = mock_classifier_instance
        mock_tracker.return_value = mock_tracker_instance
        mock_redactor.return_value = mock_redactor_instance
        
        # Create test frame
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_redactor_instance.redact_regions.return_value = test_frame
        
        # Create pipeline and process frame
        pipeline = RealtimePipeline(self.config)
        result_frame = pipeline.process_frame(test_frame, 0)
        
        # Verify processing steps were called
        mock_detector_instance.detect.assert_called_once()
        mock_tracker_instance.propagate_tracks.assert_not_called()  # No previous frame
        mock_tracker_instance.get_active_tracks.assert_called()
        mock_redactor_instance.redact_regions.assert_called_once()
        
        # Verify result
        assert result_frame is not None
        assert result_frame.shape == test_frame.shape
    
    def test_error_recovery_in_pipeline(self):
        """Test error recovery in pipeline processing."""
        # Create pipeline with real components but mock failures
        pipeline = RealtimePipeline(self.config)
        
        # Mock text detector to fail
        with patch.object(pipeline.text_detector, 'detect', side_effect=Exception("Detection failed")):
            test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Should not raise exception, should return original frame
            result_frame = pipeline.process_frame(test_frame, 0)
            
            assert result_frame is not None
            # In case of error, should return original frame
            np.testing.assert_array_equal(result_frame, test_frame)


class TestVideoTransformerIntegration:
    """Test video transformer integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        
        # Mock pipeline
        self.mock_pipeline = Mock()
        self.mock_pipeline.process_frame.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.mock_pipeline.get_stats.return_value = {
            'frames_processed': 0,
            'tracks_active': 0,
            'ocr_queue_size': 0,
            'ocr_processed': 0
        }
    
    def test_video_transformer_initialization(self):
        """Test video transformer initialization."""
        transformer = VideoTransformer(self.config, self.mock_pipeline)
        
        assert transformer.config == self.config
        assert transformer.pipeline == self.mock_pipeline
        assert transformer.frame_count == 0
        assert transformer.dropped_frames == 0
        assert transformer.total_frames == 0
    
    def test_frame_processing_performance(self):
        """Test frame processing with performance monitoring."""
        transformer = VideoTransformer(self.config, self.mock_pipeline)
        
        # Create mock frame
        mock_frame = Mock()
        mock_frame.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_frame.pts = 0
        mock_frame.time_base = None
        
        # Process multiple frames
        for i in range(10):
            with patch('privacy_redactor_rt.webrtc_utils.av.VideoFrame') as mock_av_frame:
                mock_output_frame = Mock()
                mock_av_frame.from_ndarray.return_value = mock_output_frame
                
                result = transformer.recv(mock_frame)
                
                assert result is not None
        
        # Verify statistics were updated
        stats = transformer.get_stats()
        assert stats['frames_processed'] == 10
        assert stats['frames_total'] == 10
        assert stats['fps_current'] > 0
    
    def test_backpressure_management(self):
        """Test backpressure management and frame dropping."""
        # Configure for aggressive backpressure
        self.config.realtime.backpressure_threshold_ms = 1.0  # Very low threshold
        
        transformer = VideoTransformer(self.config, self.mock_pipeline)
        
        # Mock slow processing
        def slow_process_frame(frame, frame_idx):
            time.sleep(0.01)  # Simulate slow processing
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        
        self.mock_pipeline.process_frame.side_effect = slow_process_frame
        
        # Create mock frame
        mock_frame = Mock()
        mock_frame.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_frame.pts = 0
        mock_frame.time_base = None
        
        # Process frames rapidly
        for i in range(20):
            with patch('privacy_redactor_rt.webrtc_utils.av.VideoFrame') as mock_av_frame:
                mock_output_frame = Mock()
                mock_av_frame.from_ndarray.return_value = mock_output_frame
                
                transformer.recv(mock_frame)
        
        # Should have dropped some frames due to backpressure
        stats = transformer.get_stats()
        assert stats['frames_total'] == 20
        # Some frames might be dropped depending on timing
    
    def test_performance_health_monitoring(self):
        """Test performance health monitoring."""
        transformer = VideoTransformer(self.config, self.mock_pipeline)
        
        # Initially should be healthy (no data)
        assert transformer.is_performance_healthy()
        
        # Simulate good performance
        transformer.fps_ema = 30.0
        transformer.latency_ema = 0.01  # 10ms
        transformer.dropped_frames = 0
        transformer.total_frames = 100
        
        assert transformer.is_performance_healthy()
        
        # Simulate poor performance
        transformer.fps_ema = 15.0  # Below threshold
        transformer.latency_ema = 0.2  # 200ms - above threshold
        transformer.dropped_frames = 20  # 20% drop rate
        
        assert not transformer.is_performance_healthy()


if __name__ == "__main__":
    pytest.main([__file__])