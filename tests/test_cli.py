"""Unit tests for CLI interface and offline processing."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2
from typer.testing import CliRunner

from privacy_redactor_rt.cli import app, OfflineProcessor, validate_input_file, validate_output_file, load_and_validate_config
from privacy_redactor_rt.config import Config
from privacy_redactor_rt.types import BBox, Detection, Match, Track


class TestOfflineProcessor:
    """Test cases for OfflineProcessor class."""
    
    def test_init(self):
        """Test OfflineProcessor initialization."""
        config = Config()
        processor = OfflineProcessor(config)
        
        assert processor.config == config
        assert processor.pipeline is None
        assert processor.video_source is None
        assert processor.recorder is None
        assert processor.stats['frames_processed'] == 0
    
    @patch('privacy_redactor_rt.cli.RealtimePipeline')
    @patch('privacy_redactor_rt.cli.VideoSource')
    @patch('privacy_redactor_rt.cli.MP4Recorder')
    def test_process_video_success(self, mock_recorder_class, mock_video_source_class, mock_pipeline_class):
        """Test successful video processing."""
        # Setup mocks
        config = Config()
        processor = OfflineProcessor(config)
        
        # Mock video source
        mock_video_source = Mock()
        mock_video_source.open_file.return_value = True
        mock_video_source.get_source_info.return_value = {'total_frames': 100, 'source_fps': 30}
        mock_video_source.get_frame_iterator.return_value = [
            (np.zeros((720, 1280, 3), dtype=np.uint8), 1.0, (0, 0)) for _ in range(5)
        ]
        mock_video_source_class.return_value = mock_video_source
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.process_frame.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_pipeline.get_stats.return_value = {'tracks_active': 2, 'ocr_processed': 10}
        mock_pipeline_class.return_value = mock_pipeline
        
        # Mock recorder
        mock_recorder = Mock()
        mock_recorder.start_recording.return_value = True
        mock_recorder.write_frame.return_value = True
        mock_recorder.stop_recording.return_value = {
            'duration_seconds': 5.0,
            'file_size_bytes': 1024000
        }
        mock_recorder_class.return_value = mock_recorder
        
        # Test processing
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            
            # Create dummy input file
            input_path.touch()
            
            stats = processor.process_video(input_path, output_path)
            
            # Verify calls
            mock_video_source.open_file.assert_called_once_with(input_path)
            mock_recorder.start_recording.assert_called_once_with(str(output_path))
            mock_pipeline.start.assert_called_once()
            mock_pipeline.stop.assert_called_once()
            
            # Verify stats
            assert stats['frames_processed'] == 5
            assert 'processing_time' in stats
            assert 'avg_fps' in stats
            assert stats['duration_seconds'] == 5.0
    
    @patch('privacy_redactor_rt.cli.RealtimePipeline')
    @patch('privacy_redactor_rt.cli.VideoSource')
    def test_process_video_invalid_input(self, mock_video_source_class, mock_pipeline_class):
        """Test processing with invalid input file."""
        config = Config()
        processor = OfflineProcessor(config)
        
        # Mock video source that fails to open
        mock_video_source = Mock()
        mock_video_source.open_file.return_value = False
        mock_video_source_class.return_value = mock_video_source
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "nonexistent.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            
            with pytest.raises(ValueError, match="Failed to open input video"):
                processor.process_video(input_path, output_path)


class TestValidationFunctions:
    """Test cases for validation functions."""
    
    def test_validate_input_file_success(self):
        """Test successful input file validation."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
            # Create a minimal valid video file using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(temp_path), fourcc, 20.0, (640, 480))
            
            # Write a few frames
            for _ in range(5):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                out.write(frame)
            out.release()
            
            try:
                # Should not raise any exception
                validate_input_file(temp_path)
            finally:
                temp_path.unlink()
    
    def test_validate_input_file_not_exists(self):
        """Test input file validation with non-existent file."""
        non_existent = Path("/path/that/does/not/exist.mp4")
        
        with pytest.raises(Exception):  # typer.BadParameter in real usage
            validate_input_file(non_existent)
    
    def test_validate_input_file_not_video(self):
        """Test input file validation with non-video file."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b"This is not a video file")
            
            try:
                with pytest.raises(Exception):  # typer.BadParameter in real usage
                    validate_input_file(temp_path)
            finally:
                temp_path.unlink()
    
    def test_validate_output_file_success(self):
        """Test successful output file validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.mp4"
            
            # Should not raise any exception
            validate_output_file(output_path)
            
            # Should create parent directory
            assert output_path.parent.exists()
    
    def test_validate_output_file_nested_path(self):
        """Test output file validation with nested path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subdir" / "nested" / "output.mp4"
            
            # Should not raise any exception
            validate_output_file(output_path)
            
            # Should create parent directories
            assert output_path.parent.exists()
    
    def test_load_and_validate_config_default(self):
        """Test loading default configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "nonexistent.yaml"
            
            config = load_and_validate_config(config_path)
            
            # Should return default config
            assert isinstance(config, Config)
            assert config.recording.enabled is True  # CLI override
    
    def test_load_and_validate_config_existing(self):
        """Test loading existing configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            temp_file.write("""
io:
  target_width: 1920
  target_height: 1080
detection:
  min_text_confidence: 0.8
""")
            temp_path = Path(temp_file.name)
        
        try:
            config = load_and_validate_config(temp_path)
            
            assert config.io.target_width == 1920
            assert config.io.target_height == 1080
            assert config.detection.min_text_confidence == 0.8
            assert config.recording.enabled is True  # CLI override
        finally:
            temp_path.unlink()


class TestCLICommands:
    """Test cases for CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_redact_video_help(self):
        """Test redact-video command help."""
        result = self.runner.invoke(app, ["redact-video", "--help"])
        assert result.exit_code == 0
        assert "Redact sensitive information from video file" in result.stdout
    
    def test_batch_process_help(self):
        """Test batch-process command help."""
        result = self.runner.invoke(app, ["batch-process", "--help"])
        assert result.exit_code == 0
        assert "Batch process multiple video files" in result.stdout
    
    def test_run_app_help(self):
        """Test run-app command help."""
        result = self.runner.invoke(app, ["run-app", "--help"])
        assert result.exit_code == 0
        assert "Run the Streamlit web interface" in result.stdout
    
    @patch('privacy_redactor_rt.cli.OfflineProcessor')
    @patch('privacy_redactor_rt.cli.validate_input_file')
    @patch('privacy_redactor_rt.cli.validate_output_file')
    @patch('privacy_redactor_rt.cli.load_and_validate_config')
    def test_redact_video_basic(self, mock_load_config, mock_validate_output, 
                               mock_validate_input, mock_processor_class):
        """Test basic redact-video command execution."""
        # Setup mocks
        mock_config = Config()
        mock_load_config.return_value = mock_config
        
        mock_processor = Mock()
        mock_processor.process_video.return_value = {
            'frames_processed': 100,
            'processing_time': 10.0,
            'avg_fps': 10.0,
            'file_size_bytes': 1024000,
            'duration_seconds': 10.0,
            'pipeline_stats': {'tracks_active': 5, 'ocr_processed': 20}
        }
        mock_processor_class.return_value = mock_processor
        
        # Test command
        result = self.runner.invoke(app, [
            "redact-video", 
            "input.mp4", 
            "output.mp4",
            "--no-progress",
            "--quiet"
        ])
        
        assert result.exit_code == 0
        mock_validate_input.assert_called_once()
        mock_validate_output.assert_called_once()
        mock_processor.process_video.assert_called_once()
    
    @patch('privacy_redactor_rt.cli.OfflineProcessor')
    @patch('privacy_redactor_rt.cli.validate_input_file')
    @patch('privacy_redactor_rt.cli.validate_output_file')
    @patch('privacy_redactor_rt.cli.load_and_validate_config')
    def test_redact_video_with_options(self, mock_load_config, mock_validate_output,
                                      mock_validate_input, mock_processor_class):
        """Test redact-video command with various options."""
        # Setup mocks
        mock_config = Config()
        mock_load_config.return_value = mock_config
        
        mock_processor = Mock()
        mock_processor.process_video.return_value = {
            'frames_processed': 50,
            'processing_time': 5.0,
            'avg_fps': 10.0,
            'file_size_bytes': 512000,
            'duration_seconds': 5.0,
            'pipeline_stats': {'tracks_active': 2, 'ocr_processed': 10}
        }
        mock_processor_class.return_value = mock_processor
        
        # Test command with options
        result = self.runner.invoke(app, [
            "redact-video",
            "input.mp4",
            "output.mp4",
            "--category", "phone",
            "--category", "email",
            "--confidence", "0.8",
            "--method", "pixelate",
            "--no-progress",
            "--quiet"
        ])
        
        assert result.exit_code == 0
        
        # Verify configuration was modified
        assert mock_config.classification.categories == ["phone", "email"]
        assert mock_config.detection.min_text_confidence == 0.8
        assert mock_config.ocr.min_ocr_confidence == 0.8
        assert mock_config.redaction.default_method == "pixelate"
    
    def test_redact_video_invalid_category(self):
        """Test redact-video command with invalid category."""
        result = self.runner.invoke(app, [
            "redact-video",
            "input.mp4",
            "output.mp4",
            "--category", "invalid_category",
            "--quiet"
        ])
        
        assert result.exit_code == 1
        assert "Invalid categories" in result.stdout
    
    def test_redact_video_invalid_confidence(self):
        """Test redact-video command with invalid confidence."""
        result = self.runner.invoke(app, [
            "redact-video",
            "input.mp4", 
            "output.mp4",
            "--confidence", "1.5",
            "--quiet"
        ])
        
        assert result.exit_code == 1
        assert "Confidence must be between 0.0 and 1.0" in result.stdout
    
    def test_redact_video_invalid_method(self):
        """Test redact-video command with invalid redaction method."""
        result = self.runner.invoke(app, [
            "redact-video",
            "input.mp4",
            "output.mp4", 
            "--method", "invalid_method",
            "--quiet"
        ])
        
        assert result.exit_code == 1
        assert "Method must be one of: gaussian, pixelate, solid" in result.stdout
    
    @patch('privacy_redactor_rt.cli.Path.glob')
    @patch('privacy_redactor_rt.cli.OfflineProcessor')
    @patch('privacy_redactor_rt.cli.load_and_validate_config')
    def test_batch_process_basic(self, mock_load_config, mock_processor_class, mock_glob):
        """Test basic batch-process command."""
        # Setup mocks
        mock_config = Config()
        mock_load_config.return_value = mock_config
        
        # Mock file discovery
        mock_files = [Path("file1.mp4"), Path("file2.mp4")]
        mock_glob.return_value = mock_files
        
        # Mock processor
        mock_processor = Mock()
        mock_processor.process_video.return_value = {
            'frames_processed': 100,
            'processing_time': 10.0,
            'avg_fps': 10.0
        }
        mock_processor_class.return_value = mock_processor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            result = self.runner.invoke(app, [
                "batch-process",
                str(input_dir),
                str(output_dir),
                "--continue-on-error"
            ])
            
            assert result.exit_code == 0
            # Should process each file
            assert mock_processor.process_video.call_count == len(mock_files)
    
    @patch('subprocess.run')
    def test_run_app_basic(self, mock_subprocess):
        """Test basic run-app command."""
        mock_subprocess.return_value = None
        
        result = self.runner.invoke(app, ["run-app", "--port", "8502"])
        
        # Should attempt to run streamlit
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert "streamlit" in args
        assert "--server.port" in args
        assert "8502" in args


class TestProgressReporting:
    """Test cases for progress reporting functionality."""
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        progress_updates = []
        
        def progress_callback(progress_pct, frame_idx, total_frames):
            progress_updates.append((progress_pct, frame_idx, total_frames))
        
        # Simulate progress updates
        total_frames = 100
        for frame_idx in range(0, total_frames, 10):
            progress_pct = frame_idx / total_frames
            progress_callback(progress_pct, frame_idx, total_frames)
        
        # Verify progress updates
        assert len(progress_updates) == 10
        assert progress_updates[0] == (0.0, 0, 100)
        assert progress_updates[-1] == (0.9, 90, 100)
    
    def test_progress_with_unknown_total(self):
        """Test progress handling when total frames is unknown."""
        progress_updates = []
        
        def progress_callback(progress_pct, frame_idx, total_frames):
            progress_updates.append((progress_pct, frame_idx, total_frames))
        
        # Simulate unknown total frames (0)
        total_frames = 0
        for frame_idx in range(10):
            if total_frames > 0:
                progress_pct = frame_idx / total_frames
                progress_callback(progress_pct, frame_idx, total_frames)
        
        # Should not call progress callback when total is 0
        assert len(progress_updates) == 0


class TestErrorHandling:
    """Test cases for error handling in CLI operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @patch('privacy_redactor_rt.cli.OfflineProcessor')
    @patch('privacy_redactor_rt.cli.validate_input_file')
    @patch('privacy_redactor_rt.cli.validate_output_file')
    @patch('privacy_redactor_rt.cli.load_and_validate_config')
    def test_processing_error_handling(self, mock_load_config, mock_validate_output,
                                     mock_validate_input, mock_processor_class):
        """Test error handling during video processing."""
        # Setup mocks
        mock_config = Config()
        mock_load_config.return_value = mock_config
        
        # Mock processor that raises exception
        mock_processor = Mock()
        mock_processor.process_video.side_effect = Exception("Processing failed")
        mock_processor_class.return_value = mock_processor
        
        result = self.runner.invoke(app, [
            "redact-video",
            "input.mp4",
            "output.mp4",
            "--quiet"
        ])
        
        assert result.exit_code == 1
        assert "Unexpected error" in result.stdout
    
    def test_keyboard_interrupt_handling(self):
        """Test handling of keyboard interrupt (Ctrl+C)."""
        with patch('privacy_redactor_rt.cli.OfflineProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process_video.side_effect = KeyboardInterrupt()
            mock_processor_class.return_value = mock_processor
            
            with patch('privacy_redactor_rt.cli.validate_input_file'), \
                 patch('privacy_redactor_rt.cli.validate_output_file'), \
                 patch('privacy_redactor_rt.cli.load_and_validate_config') as mock_load_config:
                
                mock_load_config.return_value = Config()
                
                result = self.runner.invoke(app, [
                    "redact-video",
                    "input.mp4",
                    "output.mp4",
                    "--quiet"
                ])
                
                assert result.exit_code == 130  # Standard exit code for SIGINT
                assert "interrupted by user" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__])