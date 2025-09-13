"""Tests for logging utilities with privacy protection."""

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from privacy_redactor_rt.logging_utils import (
    DetectionAuditor,
    PrivacyPreservingFormatter,
    setup_logging,
    mask_sensitive_text,
    create_detection_summary
)
from privacy_redactor_rt.config import LoggingConfig
from privacy_redactor_rt.types import BBox, Match, Track


class TestPrivacyPreservingFormatter:
    """Test privacy-preserving log formatter."""
    
    def test_formatter_initialization(self):
        """Test formatter initialization with custom mask chars."""
        formatter = PrivacyPreservingFormatter(mask_chars_visible=5)
        assert formatter.mask_chars_visible == 5
    
    def test_format_basic_message(self):
        """Test formatting of basic log message."""
        formatter = PrivacyPreservingFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert "Test message" in formatted


class TestMaskSensitiveText:
    """Test text masking utility function."""
    
    def test_mask_short_text(self):
        """Test masking of very short text."""
        assert mask_sensitive_text("ab") == "a*"
        assert mask_sensitive_text("a") == "*"
        assert mask_sensitive_text("") == ""
    
    def test_mask_medium_text(self):
        """Test masking of medium length text."""
        result = mask_sensitive_text("hello", chars_visible=2)
        assert result == "he*lo"
        
        result = mask_sensitive_text("testing", chars_visible=2)
        assert result == "te***ng"
    
    def test_mask_long_text(self):
        """Test masking of long text."""
        text = "4532-1234-5678-9012"
        result = mask_sensitive_text(text, chars_visible=3)
        assert result == "453*************012"
        assert len(result) == len(text)
    
    def test_mask_default_chars_visible(self):
        """Test default number of visible characters."""
        text = "sensitive_data_here"
        result = mask_sensitive_text(text)
        assert result.startswith("sen")
        assert result.endswith("ere")
        assert "*" in result
    
    def test_mask_edge_cases(self):
        """Test edge cases for text masking."""
        # Text exactly 2 * chars_visible
        result = mask_sensitive_text("123456", chars_visible=3)
        assert result == "123456"  # No masking needed
        
        # Text just over threshold
        result = mask_sensitive_text("1234567", chars_visible=3)
        assert result == "123*567"


class TestDetectionAuditor:
    """Test detection auditor functionality."""
    
    @pytest.fixture
    def temp_audit_file(self):
        """Create temporary audit file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def logging_config(self, temp_audit_file):
        """Create logging configuration for testing."""
        return LoggingConfig(
            audit_detections=True,
            audit_file=temp_audit_file,
            log_text_previews=True,
            mask_text=True,
            mask_chars_visible=3
        )
    
    @pytest.fixture
    def sample_track(self):
        """Create sample track for testing."""
        bbox = BBox(x1=100, y1=200, x2=300, y2=250, confidence=0.85)
        match = Match(
            category="credit_card",
            confidence=0.92,
            masked_text="4532-****-****-9012",
            bbox=bbox
        )
        
        track = Track(
            id="track_001",
            bbox=bbox,
            matches=[match],
            age=5,
            hits=4,
            last_ocr_frame=10
        )
        return track
    
    def test_auditor_initialization(self, logging_config):
        """Test auditor initialization."""
        auditor = DetectionAuditor(logging_config)
        assert auditor.config == logging_config
        assert auditor.audit_file is not None
        auditor.close()
    
    def test_auditor_disabled(self):
        """Test auditor when audit is disabled."""
        config = LoggingConfig(audit_detections=False)
        auditor = DetectionAuditor(config)
        assert auditor.audit_file is None
        auditor.close()
    
    def test_mask_text_enabled(self, logging_config):
        """Test text masking when enabled."""
        auditor = DetectionAuditor(logging_config)
        
        masked = auditor._mask_text("4532-1234-5678-9012")
        assert masked == "453*************012"
        
        auditor.close()
    
    def test_mask_text_disabled(self, temp_audit_file):
        """Test text masking when disabled."""
        config = LoggingConfig(
            audit_detections=True,
            audit_file=temp_audit_file,
            mask_text=False
        )
        auditor = DetectionAuditor(config)
        
        original_text = "4532-1234-5678-9012"
        masked = auditor._mask_text(original_text)
        assert masked == original_text
        
        auditor.close()
    
    def test_prepare_detection_event(self, logging_config, sample_track):
        """Test preparation of detection event data."""
        auditor = DetectionAuditor(logging_config)
        timestamp = datetime(2024, 1, 15, 12, 30, 45)
        
        event_data = auditor._prepare_detection_event(sample_track, 42, timestamp)
        
        assert event_data['timestamp'] == '2024-01-15T12:30:45Z'
        assert event_data['frame_idx'] == 42
        assert event_data['track_id'] == 'track_001'
        assert event_data['category'] == 'credit_card'
        assert event_data['confidence'] == 0.92
        
        # Check bbox data
        bbox_data = event_data['bbox']
        assert bbox_data['x1'] == 100
        assert bbox_data['y1'] == 200
        assert bbox_data['x2'] == 300
        assert bbox_data['y2'] == 250
        assert bbox_data['width'] == 200
        assert bbox_data['height'] == 50
        
        # Check track stats
        stats = event_data['track_stats']
        assert stats['age'] == 5
        assert stats['hits'] == 4
        assert stats['hit_rate'] == 0.8
        
        # Check text preview is masked
        assert 'text_preview' in event_data
        assert event_data['text_preview'] == "453*************012"
        
        # Check all matches
        assert len(event_data['all_matches']) == 1
        match_data = event_data['all_matches'][0]
        assert match_data['category'] == 'credit_card'
        assert match_data['confidence'] == 0.92
        
        auditor.close()
    
    def test_prepare_detection_event_no_text_preview(self, temp_audit_file, sample_track):
        """Test event preparation with text previews disabled."""
        config = LoggingConfig(
            audit_detections=True,
            audit_file=temp_audit_file,
            log_text_previews=False
        )
        auditor = DetectionAuditor(config)
        
        event_data = auditor._prepare_detection_event(sample_track, 42)
        
        assert 'text_preview' not in event_data
        assert len(event_data['all_matches']) == 1
        assert 'text_preview' not in event_data['all_matches'][0]
        
        auditor.close()
    
    def test_log_detection(self, logging_config, sample_track, temp_audit_file):
        """Test logging of detection event."""
        auditor = DetectionAuditor(logging_config)
        timestamp = datetime(2024, 1, 15, 12, 30, 45)
        
        auditor.log_detection(sample_track, 42, timestamp)
        auditor.close()
        
        # Read and verify the logged data
        with open(temp_audit_file, 'r') as f:
            line = f.readline().strip()
            event_data = json.loads(line)
        
        assert event_data['timestamp'] == '2024-01-15T12:30:45Z'
        assert event_data['frame_idx'] == 42
        assert event_data['track_id'] == 'track_001'
        assert event_data['category'] == 'credit_card'
    
    def test_log_batch_detections(self, logging_config, temp_audit_file):
        """Test batch logging of multiple detections."""
        # Create multiple tracks
        tracks = []
        for i in range(3):
            bbox = BBox(x1=i*100, y1=i*50, x2=(i+1)*100, y2=(i+1)*50, confidence=0.8)
            match = Match(
                category="phone",
                confidence=0.85,
                masked_text=f"555-***-{i:04d}",
                bbox=bbox
            )
            track = Track(
                id=f"track_{i:03d}",
                bbox=bbox,
                matches=[match],
                age=i+1,
                hits=i+1,
                last_ocr_frame=i*5
            )
            tracks.append(track)
        
        auditor = DetectionAuditor(logging_config)
        timestamp = datetime(2024, 1, 15, 12, 30, 45)
        
        auditor.log_batch_detections(tracks, 100, timestamp)
        auditor.close()
        
        # Read and verify all logged events
        with open(temp_audit_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 3
        
        for i, line in enumerate(lines):
            event_data = json.loads(line.strip())
            assert event_data['track_id'] == f'track_{i:03d}'
            assert event_data['category'] == 'phone'
            assert event_data['frame_idx'] == 100
    
    def test_log_performance_stats(self, logging_config, temp_audit_file):
        """Test logging of performance statistics."""
        auditor = DetectionAuditor(logging_config)
        timestamp = datetime(2024, 1, 15, 12, 30, 45)
        
        stats = {
            'fps': 29.5,
            'latency_ms': 85.2,
            'queue_size': 2,
            'memory_mb': 156.7
        }
        
        auditor.log_performance_stats(stats, 200, timestamp)
        auditor.close()
        
        # Read and verify the logged stats
        with open(temp_audit_file, 'r') as f:
            line = f.readline().strip()
            event_data = json.loads(line)
        
        assert event_data['event_type'] == 'performance_stats'
        assert event_data['frame_idx'] == 200
        assert event_data['stats']['fps'] == 29.5
        assert event_data['stats']['latency_ms'] == 85.2
    
    def test_context_manager(self, logging_config):
        """Test auditor as context manager."""
        with DetectionAuditor(logging_config) as auditor:
            assert auditor.audit_file is not None
        
        # File should be closed after context exit
        assert auditor.audit_file is None
    
    def test_file_creation_error(self):
        """Test handling of file creation errors."""
        config = LoggingConfig(
            audit_detections=True,
            audit_file="/invalid/path/audit.jsonl"
        )
        
        with patch('privacy_redactor_rt.logging_utils.logging.error') as mock_error:
            auditor = DetectionAuditor(config)
            assert auditor.audit_file is None
            mock_error.assert_called_once()
            auditor.close()


class TestSetupLogging:
    """Test logging setup functionality."""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        config = LoggingConfig(
            log_level="INFO",
            audit_detections=False
        )
        
        auditor = setup_logging(config)
        assert isinstance(auditor, DetectionAuditor)
        auditor.close()
    
    def test_setup_logging_with_no_log_text_flag(self):
        """Test logging setup with no_log_text flag."""
        config = LoggingConfig(
            log_text_previews=True,
            audit_detections=False
        )
        
        auditor = setup_logging(config, no_log_text=True)
        assert not auditor.config.log_text_previews
        auditor.close()
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file handler."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.log') as f:
            log_file = f.name
        
        try:
            config = LoggingConfig(
                log_level="DEBUG",
                log_file=log_file,
                audit_detections=False
            )
            
            auditor = setup_logging(config)
            
            # Test that logging works
            logging.info("Test message")
            
            # Verify log file was created and contains content
            assert Path(log_file).exists()
            
            auditor.close()
        finally:
            Path(log_file).unlink(missing_ok=True)


class TestCreateDetectionSummary:
    """Test detection summary creation."""
    
    def test_empty_tracks(self):
        """Test summary with no tracks."""
        summary = create_detection_summary([])
        assert summary == {}
    
    def test_tracks_without_matches(self):
        """Test summary with tracks that have no matches."""
        bbox = BBox(x1=0, y1=0, x2=100, y2=50, confidence=0.8)
        track = Track(
            id="track_001",
            bbox=bbox,
            matches=[],
            age=1,
            hits=1,
            last_ocr_frame=0
        )
        
        summary = create_detection_summary([track])
        assert summary == {}
    
    def test_single_category_summary(self):
        """Test summary with single category."""
        bbox = BBox(x1=0, y1=0, x2=100, y2=50, confidence=0.8)
        
        matches = [
            Match(category="phone", confidence=0.9, masked_text="555-***-1234", bbox=bbox),
            Match(category="phone", confidence=0.85, masked_text="555-***-5678", bbox=bbox)
        ]
        
        track = Track(
            id="track_001",
            bbox=bbox,
            matches=matches,
            age=2,
            hits=2,
            last_ocr_frame=5
        )
        
        summary = create_detection_summary([track])
        assert summary == {"phone": 2}
    
    def test_multiple_categories_summary(self):
        """Test summary with multiple categories."""
        bbox = BBox(x1=0, y1=0, x2=100, y2=50, confidence=0.8)
        
        # First track with phone and email
        track1_matches = [
            Match(category="phone", confidence=0.9, masked_text="555-***-1234", bbox=bbox),
            Match(category="email", confidence=0.85, masked_text="use***@example.com", bbox=bbox)
        ]
        track1 = Track(
            id="track_001",
            bbox=bbox,
            matches=track1_matches,
            age=2,
            hits=2,
            last_ocr_frame=5
        )
        
        # Second track with credit card
        track2_matches = [
            Match(category="credit_card", confidence=0.92, masked_text="453*-****-****-9012", bbox=bbox)
        ]
        track2 = Track(
            id="track_002",
            bbox=bbox,
            matches=track2_matches,
            age=1,
            hits=1,
            last_ocr_frame=3
        )
        
        summary = create_detection_summary([track1, track2])
        expected = {"phone": 1, "email": 1, "credit_card": 1}
        assert summary == expected
    
    def test_multiple_tracks_same_category(self):
        """Test summary with multiple tracks of same category."""
        bbox = BBox(x1=0, y1=0, x2=100, y2=50, confidence=0.8)
        
        tracks = []
        for i in range(3):
            match = Match(
                category="api_key",
                confidence=0.9,
                masked_text=f"sk-***{i}",
                bbox=bbox
            )
            track = Track(
                id=f"track_{i:03d}",
                bbox=bbox,
                matches=[match],
                age=1,
                hits=1,
                last_ocr_frame=i
            )
            tracks.append(track)
        
        summary = create_detection_summary(tracks)
        assert summary == {"api_key": 3}


class TestIntegration:
    """Integration tests for logging system."""
    
    def test_end_to_end_logging_workflow(self):
        """Test complete logging workflow from setup to audit."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as f:
            audit_file = f.name
        
        try:
            # Setup logging
            config = LoggingConfig(
                log_level="INFO",
                audit_detections=True,
                audit_file=audit_file,
                log_text_previews=True,
                mask_text=True,
                mask_chars_visible=2
            )
            
            auditor = setup_logging(config)
            
            # Create sample detection
            bbox = BBox(x1=50, y1=100, x2=200, y2=150, confidence=0.88)
            match = Match(
                category="credit_card",
                confidence=0.94,
                masked_text="4532-1234-5678-9012",
                bbox=bbox
            )
            track = Track(
                id="integration_test_track",
                bbox=bbox,
                matches=[match],
                age=3,
                hits=3,
                last_ocr_frame=15
            )
            
            # Log the detection
            timestamp = datetime(2024, 1, 15, 14, 30, 0)
            auditor.log_detection(track, 150, timestamp)
            
            # Log performance stats
            stats = {'fps': 30.0, 'latency_ms': 75.5}
            auditor.log_performance_stats(stats, 150, timestamp)
            
            auditor.close()
            
            # Verify logged data
            with open(audit_file, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 2
            
            # Check detection event
            detection_event = json.loads(lines[0])
            assert detection_event['track_id'] == 'integration_test_track'
            assert detection_event['category'] == 'credit_card'
            assert detection_event['text_preview'] == '45***************12'
            
            # Check performance event
            perf_event = json.loads(lines[1])
            assert perf_event['event_type'] == 'performance_stats'
            assert perf_event['stats']['fps'] == 30.0
            
        finally:
            Path(audit_file).unlink(missing_ok=True)
    
    def test_privacy_protection_no_log_text_flag(self):
        """Test that --no-log-text flag completely disables text logging."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as f:
            audit_file = f.name
        
        try:
            config = LoggingConfig(
                audit_detections=True,
                audit_file=audit_file,
                log_text_previews=True,  # This should be overridden
                mask_text=True
            )
            
            # Setup with no_log_text=True
            auditor = setup_logging(config, no_log_text=True)
            
            # Create and log detection
            bbox = BBox(x1=0, y1=0, x2=100, y2=50, confidence=0.9)
            match = Match(
                category="phone",
                confidence=0.95,
                masked_text="555-123-4567",
                bbox=bbox
            )
            track = Track(
                id="privacy_test_track",
                bbox=bbox,
                matches=[match],
                age=1,
                hits=1,
                last_ocr_frame=0
            )
            
            auditor.log_detection(track, 100)
            auditor.close()
            
            # Verify no text previews in logged data
            with open(audit_file, 'r') as f:
                event_data = json.loads(f.readline())
            
            assert 'text_preview' not in event_data
            assert len(event_data['all_matches']) == 1
            assert 'text_preview' not in event_data['all_matches'][0]
            
        finally:
            Path(audit_file).unlink(missing_ok=True)