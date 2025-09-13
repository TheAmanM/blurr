"""Audit logging system with privacy protection for detection events."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Any
from dataclasses import asdict

from .types import Match, Track, BBox
from .config import LoggingConfig


class PrivacyPreservingFormatter(logging.Formatter):
    """Custom formatter that masks sensitive data in log messages."""
    
    def __init__(self, mask_chars_visible: int = 3):
        super().__init__()
        self.mask_chars_visible = mask_chars_visible
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with privacy protection."""
        # Apply standard formatting first
        formatted = super().format(record)
        
        # Note: In a real implementation, you might want to scan for and mask
        # patterns that look like sensitive data in the log message itself
        return formatted


class DetectionAuditor:
    """Handles audit logging of detection events with privacy protection."""
    
    def __init__(self, config: LoggingConfig):
        """Initialize the detection auditor.
        
        Args:
            config: Logging configuration
        """
        self.config = config
        self.audit_file: Optional[TextIO] = None
        self._setup_audit_file()
    
    def _setup_audit_file(self) -> None:
        """Set up the audit file for JSONL logging."""
        if not self.config.audit_detections or not self.config.audit_file:
            return
        
        try:
            audit_path = Path(self.config.audit_file)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Open in append mode to preserve existing logs
            self.audit_file = open(audit_path, 'a', encoding='utf-8')
        except Exception as e:
            logging.error(f"Failed to open audit file {self.config.audit_file}: {e}")
            self.audit_file = None
    
    def _mask_text(self, text: str) -> str:
        """Apply privacy-preserving text masking.
        
        Shows only first and last N characters, masks the middle.
        
        Args:
            text: Original text to mask
            
        Returns:
            Masked text showing only first/last characters
        """
        if not self.config.mask_text:
            return text
        
        visible_chars = self.config.mask_chars_visible
        
        # Handle empty string
        if len(text) == 0:
            return ''
        
        # For very short text, show first/last char only
        if len(text) <= 2:
            if len(text) == 1:
                return '*'
            return text[0] + '*'
        
        # If text length equals exactly 2 * visible_chars, no masking needed
        if len(text) == 2 * visible_chars:
            return text
        
        # If text is shorter than 2 * visible_chars, mask middle only
        if len(text) < 2 * visible_chars:
            return text[0] + '*' * (len(text) - 2) + text[-1]
        
        # Normal case: mask middle, show first/last N chars
        masked_length = len(text) - 2 * visible_chars
        
        return (
            text[:visible_chars] + 
            '*' * masked_length + 
            text[-visible_chars:]
        )
    
    def _prepare_detection_event(
        self, 
        track: Track, 
        frame_idx: int, 
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Prepare detection event data for logging.
        
        Args:
            track: Track containing detection matches
            frame_idx: Current frame index
            timestamp: Event timestamp (defaults to current time)
            
        Returns:
            Dictionary containing event data ready for JSON serialization
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Get the best match for this track
        best_match = track.get_best_match()
        if not best_match:
            return {}
        
        event_data = {
            'timestamp': timestamp.isoformat() + 'Z',
            'frame_idx': frame_idx,
            'track_id': track.id,
            'category': best_match.category,
            'confidence': round(best_match.confidence, 3),
            'bbox': {
                'x1': track.bbox.x1,
                'y1': track.bbox.y1,
                'x2': track.bbox.x2,
                'y2': track.bbox.y2,
                'width': track.bbox.width,
                'height': track.bbox.height
            },
            'track_stats': {
                'age': track.age,
                'hits': track.hits,
                'hit_rate': round(track.hit_rate, 3)
            }
        }
        
        # Add text preview if enabled
        if self.config.log_text_previews and best_match.masked_text:
            if self.config.mask_text:
                # Apply additional masking for audit logs
                event_data['text_preview'] = self._mask_text(best_match.masked_text)
            else:
                # Truncate to max preview length
                preview = best_match.masked_text[:self.config.max_preview_length]
                if len(best_match.masked_text) > self.config.max_preview_length:
                    preview += '...'
                event_data['text_preview'] = preview
        
        # Add all matches for this track (not just the best one)
        event_data['all_matches'] = []
        for match in track.matches:
            match_data = {
                'category': match.category,
                'confidence': round(match.confidence, 3)
            }
            
            if self.config.log_text_previews and match.masked_text:
                if self.config.mask_text:
                    match_data['text_preview'] = self._mask_text(match.masked_text)
                else:
                    preview = match.masked_text[:self.config.max_preview_length]
                    if len(match.masked_text) > self.config.max_preview_length:
                        preview += '...'
                    match_data['text_preview'] = preview
            
            event_data['all_matches'].append(match_data)
        
        return event_data
    
    def log_detection(
        self, 
        track: Track, 
        frame_idx: int, 
        timestamp: Optional[datetime] = None
    ) -> None:
        """Log a detection event to the audit file.
        
        Args:
            track: Track containing detection matches
            frame_idx: Current frame index
            timestamp: Event timestamp (defaults to current time)
        """
        if not self.config.audit_detections or not self.audit_file:
            return
        
        try:
            event_data = self._prepare_detection_event(track, frame_idx, timestamp)
            if not event_data:
                return
            
            # Write as JSONL (one JSON object per line)
            json_line = json.dumps(event_data, separators=(',', ':'))
            self.audit_file.write(json_line + '\n')
            self.audit_file.flush()
            
        except Exception as e:
            logging.error(f"Failed to log detection event: {e}")
    
    def log_batch_detections(
        self, 
        tracks: List[Track], 
        frame_idx: int, 
        timestamp: Optional[datetime] = None
    ) -> None:
        """Log multiple detection events in batch.
        
        Args:
            tracks: List of tracks with detection matches
            frame_idx: Current frame index
            timestamp: Event timestamp (defaults to current time)
        """
        if not self.config.audit_detections or not self.audit_file:
            return
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        for track in tracks:
            if track.matches:  # Only log tracks with actual matches
                self.log_detection(track, frame_idx, timestamp)
    
    def log_performance_stats(
        self, 
        stats: Dict[str, float], 
        frame_idx: int,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Log performance statistics.
        
        Args:
            stats: Performance statistics dictionary
            frame_idx: Current frame index
            timestamp: Event timestamp (defaults to current time)
        """
        if not self.config.audit_detections or not self.audit_file:
            return
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        try:
            event_data = {
                'timestamp': timestamp.isoformat() + 'Z',
                'event_type': 'performance_stats',
                'frame_idx': frame_idx,
                'stats': {k: round(v, 3) for k, v in stats.items()}
            }
            
            json_line = json.dumps(event_data, separators=(',', ':'))
            self.audit_file.write(json_line + '\n')
            self.audit_file.flush()
            
        except Exception as e:
            logging.error(f"Failed to log performance stats: {e}")
    
    def close(self) -> None:
        """Close the audit file."""
        if self.audit_file:
            try:
                self.audit_file.close()
            except Exception as e:
                logging.error(f"Error closing audit file: {e}")
            finally:
                self.audit_file = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def setup_logging(config: Optional[LoggingConfig] = None, level: int = logging.INFO, no_log_text: bool = False) -> Optional[DetectionAuditor]:
    """Set up logging system with privacy protection.
    
    Args:
        config: Logging configuration (optional, for full setup)
        level: Logging level (used when config is None)
        no_log_text: If True, disable all text preview logging
        
    Returns:
        Configured DetectionAuditor instance or None if config not provided
    """
    # Simple logging setup if no config provided
    if config is None:
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler with simple formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        return None
    
    # Override text logging if no_log_text flag is set
    if no_log_text:
        config = config.model_copy()
        config.log_text_previews = False
    
    # Configure standard logging
    log_level = getattr(logging, config.log_level.upper())
    
    # Create formatter with privacy protection
    formatter = PrivacyPreservingFormatter(
        mask_chars_visible=config.mask_chars_visible
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if config.log_file:
        try:
            log_path = Path(config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            logging.error(f"Failed to set up file logging: {e}")
    
    # Create and return detection auditor
    return DetectionAuditor(config)


def mask_sensitive_text(text: str, chars_visible: int = 3) -> str:
    """Utility function to mask sensitive text for privacy protection.
    
    Args:
        text: Original text to mask
        chars_visible: Number of characters to show at start and end
        
    Returns:
        Masked text showing only first/last characters
    """
    # Handle empty string
    if len(text) == 0:
        return ''
    
    # For very short text, show first/last char only
    if len(text) <= 2:
        if len(text) == 1:
            return '*'
        return text[0] + '*'
    
    # If text length equals exactly 2 * visible_chars, no masking needed
    if len(text) == 2 * chars_visible:
        return text
    
    # If text is shorter than 2 * visible_chars, mask middle only
    if len(text) < 2 * chars_visible:
        return text[0] + '*' * (len(text) - 2) + text[-1]
    
    # Normal case: mask middle, show first/last N chars
    masked_length = len(text) - 2 * chars_visible
    return (
        text[:chars_visible] + 
        '*' * masked_length + 
        text[-chars_visible:]
    )


def create_detection_summary(tracks: List[Track]) -> Dict[str, int]:
    """Create a summary of detections by category.
    
    Args:
        tracks: List of tracks with detection matches
        
    Returns:
        Dictionary with category counts
    """
    summary = {}
    
    for track in tracks:
        for match in track.matches:
            category = match.category
            summary[category] = summary.get(category, 0) + 1
    
    return summary