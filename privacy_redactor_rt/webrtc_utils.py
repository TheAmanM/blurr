"""WebRTC integration with performance monitoring."""

import time
import logging
from typing import Dict, Optional, Any
from collections import deque
import numpy as np

# Handle optional dependencies for testing
try:
    import av
    import cv2
    from streamlit_webrtc import VideoTransformerBase
    HAS_WEBRTC_DEPS = True
except ImportError:
    # Create mock base class and types for testing
    class VideoTransformerBase:
        pass
    
    # Mock av module for type annotations
    class MockAV:
        class VideoFrame:
            pass
    
    av = MockAV()
    HAS_WEBRTC_DEPS = False

from .pipeline import RealtimePipeline
from .config import Config

logger = logging.getLogger(__name__)


class VideoTransformer(VideoTransformerBase):
    """WebRTC video transformer with performance monitoring and backpressure management."""
    
    def __init__(self, config: Config, pipeline: RealtimePipeline, event_callback=None):
        """Initialize video transformer with performance monitoring.
        
        Args:
            config: Complete configuration object
            pipeline: Real-time processing pipeline
            event_callback: Optional callback for detection events
        """
        super().__init__()
        self.config = config
        self.pipeline = pipeline
        self.event_callback = event_callback
        
        # Frame processing state
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.target_frame_interval = 1.0 / config.io.target_fps
        
        # Performance monitoring with EMA
        self.fps_window = deque(maxlen=config.performance.stats_window_size)
        self.latency_window = deque(maxlen=config.performance.stats_window_size)
        self.processing_times = deque(maxlen=config.performance.stats_window_size)
        
        # EMA smoothing factors
        self.fps_ema = 0.0
        self.latency_ema = 0.0
        self.processing_ema = 0.0
        self.ema_alpha = 0.1
        
        # Backpressure management
        self.dropped_frames = 0
        self.total_frames = 0
        self.last_drop_time = 0.0
        self.consecutive_slow_frames = 0
        
        # Frame normalization cache
        self._letterbox_cache: Optional[Dict[str, Any]] = None
        
        logger.info(f"VideoTransformer initialized with target {config.io.target_fps} FPS")
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming WebRTC frame with performance monitoring and backpressure.
        
        Args:
            frame: Input video frame from WebRTC
            
        Returns:
            Processed video frame with redactions applied
        """
        start_time = time.time()
        self.total_frames += 1
        
        try:
            # Convert av.VideoFrame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Check if we should drop this frame for performance
            if self._should_drop_frame(start_time):
                self.dropped_frames += 1
                logger.debug(f"Dropped frame {self.frame_count} for backpressure management")
                # Return previous frame or black frame
                return self._create_fallback_frame(frame, img)
            
            # Normalize frame to target resolution
            normalized_img = self._normalize_frame(img)
            
            # Process through pipeline
            processing_start = time.time()
            processed_img = self.pipeline.process_frame(normalized_img, self.frame_count)
            processing_time = time.time() - processing_start
            
            # Report detection events if callback is available
            if self.event_callback:
                self._report_detection_events()
            
            # Update performance statistics
            self._update_performance_stats(start_time, processing_time)
            
            # Convert back to av.VideoFrame
            if HAS_WEBRTC_DEPS:
                output_frame = av.VideoFrame.from_ndarray(processed_img, format="bgr24")
                output_frame.pts = frame.pts
                output_frame.time_base = frame.time_base
            else:
                # For testing, return a mock frame
                output_frame = type('MockFrame', (), {
                    'pts': getattr(frame, 'pts', 0),
                    'time_base': getattr(frame, 'time_base', None),
                    'data': processed_img
                })()
            
            self.frame_count += 1
            return output_frame
            
        except Exception as e:
            logger.error(f"Error processing WebRTC frame {self.frame_count}: {e}")
            import traceback
            traceback.print_exc()
            # Return original frame on error
            return frame
    
    def _should_drop_frame(self, current_time: float) -> bool:
        """Determine if frame should be dropped for backpressure management.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if frame should be dropped
        """
        # Check if we're processing too slowly
        time_since_last = current_time - self.last_frame_time
        
        # If we're significantly behind target FPS, consider dropping
        if time_since_last < self.target_frame_interval * 0.5:
            # Frames coming too fast, drop some
            return True
        
        # Check recent processing performance
        if len(self.processing_times) > 5:
            recent_avg = sum(list(self.processing_times)[-5:]) / 5
            if recent_avg > self.config.realtime.backpressure_threshold_ms / 1000.0:
                self.consecutive_slow_frames += 1
                # Drop every other frame if consistently slow
                if self.consecutive_slow_frames > 3 and self.frame_count % 2 == 0:
                    return True
            else:
                self.consecutive_slow_frames = 0
        
        # Check if current FPS is below minimum threshold
        if self.fps_ema > 0 and self.fps_ema < self.config.realtime.min_fps_threshold:
            # Drop frames more aggressively
            drop_probability = 1.0 - (self.fps_ema / self.config.realtime.min_fps_threshold)
            if np.random.random() < drop_probability:
                return True
        
        return False
    
    def _create_fallback_frame(self, original_frame: av.VideoFrame, img: np.ndarray) -> av.VideoFrame:
        """Create fallback frame when dropping for performance.
        
        Args:
            original_frame: Original WebRTC frame
            img: Original image array
            
        Returns:
            Fallback frame (black or previous frame)
        """
        try:
            # Create black frame of same dimensions
            black_img = np.zeros_like(img)
            if HAS_WEBRTC_DEPS:
                fallback_frame = av.VideoFrame.from_ndarray(black_img, format="bgr24")
                fallback_frame.pts = original_frame.pts
                fallback_frame.time_base = original_frame.time_base
                return fallback_frame
            else:
                # For testing, return a mock frame
                return type('MockFrame', (), {
                    'pts': getattr(original_frame, 'pts', 0),
                    'time_base': getattr(original_frame, 'time_base', None),
                    'data': black_img
                })()
        except Exception:
            # If that fails, return original frame
            return original_frame
    
    def _normalize_frame(self, img: np.ndarray) -> np.ndarray:
        """Normalize frame to target resolution with letterboxing.
        
        Args:
            img: Input image array
            
        Returns:
            Normalized image at target resolution
        """
        target_h, target_w = self.config.io.target_height, self.config.io.target_width
        
        # Check if we can use cached letterbox parameters
        h, w = img.shape[:2]
        cache_key = f"{w}x{h}"
        
        if self._letterbox_cache is None or self._letterbox_cache.get('key') != cache_key:
            # Calculate letterbox parameters
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Calculate padding
            pad_x = (target_w - new_w) // 2
            pad_y = (target_h - new_h) // 2
            
            self._letterbox_cache = {
                'key': cache_key,
                'scale': scale,
                'new_size': (new_w, new_h),
                'padding': (pad_x, pad_y)
            }
        
        cache = self._letterbox_cache
        
        # Resize image
        if cache['scale'] != 1.0:
            if HAS_WEBRTC_DEPS:
                resized = cv2.resize(img, cache['new_size'], interpolation=cv2.INTER_LINEAR)
            else:
                # Fallback for testing - simple crop/pad to target size
                new_w, new_h = cache['new_size']
                h, w = img.shape[:2]
                
                if new_h <= h and new_w <= w:
                    # Crop if target is smaller
                    resized = img[:new_h, :new_w]
                else:
                    # Pad if target is larger
                    resized = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    copy_h = min(h, new_h)
                    copy_w = min(w, new_w)
                    resized[:copy_h, :copy_w] = img[:copy_h, :copy_w]
        else:
            resized = img
        
        # Apply letterboxing if needed
        if cache['padding'][0] > 0 or cache['padding'][1] > 0:
            letterboxed = np.full(
                (target_h, target_w, 3), 
                self.config.io.letterbox_color, 
                dtype=np.uint8
            )
            
            pad_x, pad_y = cache['padding']
            new_h, new_w = cache['new_size']
            
            # Ensure resized image fits in letterboxed frame
            end_y = min(pad_y + new_h, target_h)
            end_x = min(pad_x + new_w, target_w)
            actual_h = end_y - pad_y
            actual_w = end_x - pad_x
            
            letterboxed[pad_y:end_y, pad_x:end_x] = resized[:actual_h, :actual_w]
            return letterboxed
        
        return resized
    
    def _update_performance_stats(self, frame_start_time: float, processing_time: float) -> None:
        """Update performance statistics with EMA smoothing.
        
        Args:
            frame_start_time: When frame processing started
            processing_time: Time spent in pipeline processing
        """
        current_time = time.time()
        
        # Calculate frame-to-frame interval for FPS
        if self.last_frame_time > 0:
            frame_interval = current_time - self.last_frame_time
            current_fps = 1.0 / frame_interval if frame_interval > 0 else 0.0
            
            # Update FPS with EMA
            if self.fps_ema == 0:
                self.fps_ema = current_fps
            else:
                self.fps_ema = self.ema_alpha * current_fps + (1 - self.ema_alpha) * self.fps_ema
            
            self.fps_window.append(current_fps)
        
        # Calculate total latency (frame start to completion)
        total_latency = current_time - frame_start_time
        
        # Update latency with EMA
        if self.latency_ema == 0:
            self.latency_ema = total_latency
        else:
            self.latency_ema = self.ema_alpha * total_latency + (1 - self.ema_alpha) * self.latency_ema
        
        self.latency_window.append(total_latency)
        
        # Update processing time with EMA
        if self.processing_ema == 0:
            self.processing_ema = processing_time
        else:
            self.processing_ema = self.ema_alpha * processing_time + (1 - self.ema_alpha) * self.processing_ema
        
        self.processing_times.append(processing_time)
        
        self.last_frame_time = current_time
    
    def get_stats(self) -> Dict[str, float]:
        """Get comprehensive performance statistics.
        
        Returns:
            Dictionary with performance metrics including EMA values
        """
        # Calculate window-based statistics
        fps_avg = np.mean(self.fps_window) if self.fps_window else 0.0
        fps_std = np.std(self.fps_window) if len(self.fps_window) > 1 else 0.0
        
        latency_avg = np.mean(self.latency_window) if self.latency_window else 0.0
        latency_std = np.std(self.latency_window) if len(self.latency_window) > 1 else 0.0
        latency_p95 = np.percentile(self.latency_window, 95) if self.latency_window else 0.0
        
        processing_avg = np.mean(self.processing_times) if self.processing_times else 0.0
        processing_std = np.std(self.processing_times) if len(self.processing_times) > 1 else 0.0
        
        # Calculate drop rate
        drop_rate = (self.dropped_frames / max(1, self.total_frames)) * 100
        
        # Get pipeline statistics
        pipeline_stats = self.pipeline.get_stats()
        
        return {
            # Frame processing stats
            'fps_current': self.fps_ema,
            'fps_avg': fps_avg,
            'fps_std': fps_std,
            'fps_target': self.config.io.target_fps,
            
            # Latency stats (in milliseconds)
            'latency_current_ms': self.latency_ema * 1000,
            'latency_avg_ms': latency_avg * 1000,
            'latency_std_ms': latency_std * 1000,
            'latency_p95_ms': latency_p95 * 1000,
            
            # Processing time stats (in milliseconds)
            'processing_current_ms': self.processing_ema * 1000,
            'processing_avg_ms': processing_avg * 1000,
            'processing_std_ms': processing_std * 1000,
            
            # Frame management
            'frames_total': self.total_frames,
            'frames_processed': self.frame_count,
            'frames_dropped': self.dropped_frames,
            'drop_rate_percent': drop_rate,
            
            # Backpressure indicators
            'consecutive_slow_frames': self.consecutive_slow_frames,
            'backpressure_threshold_ms': self.config.realtime.backpressure_threshold_ms,
            
            # Pipeline stats
            **pipeline_stats
        }
    
    def reset_stats(self) -> None:
        """Reset all performance statistics."""
        self.fps_window.clear()
        self.latency_window.clear()
        self.processing_times.clear()
        
        self.fps_ema = 0.0
        self.latency_ema = 0.0
        self.processing_ema = 0.0
        
        self.dropped_frames = 0
        self.total_frames = 0
        self.frame_count = 0
        self.consecutive_slow_frames = 0
        
        self.last_frame_time = time.time()
        
        logger.info("VideoTransformer statistics reset")
    
    def get_performance_summary(self) -> str:
        """Get human-readable performance summary.
        
        Returns:
            Formatted performance summary string
        """
        stats = self.get_stats()
        
        return f"""Performance Summary:
FPS: {stats['fps_current']:.1f} (target: {stats['fps_target']})
Latency: {stats['latency_current_ms']:.1f}ms (avg: {stats['latency_avg_ms']:.1f}ms)
Processing: {stats['processing_current_ms']:.1f}ms (avg: {stats['processing_avg_ms']:.1f}ms)
Frames: {stats['frames_processed']}/{stats['frames_total']} (dropped: {stats['drop_rate_percent']:.1f}%)
Active Tracks: {stats.get('tracks_active', 0)}
OCR Queue: {stats.get('ocr_queue_size', 0)}/{self.config.realtime.max_queue}"""
    
    def is_performance_healthy(self) -> bool:
        """Check if performance is within acceptable thresholds.
        
        Returns:
            True if performance is healthy
        """
        stats = self.get_stats()
        
        # Check FPS is above minimum threshold
        if stats['fps_current'] < self.config.realtime.min_fps_threshold:
            return False
        
        # Check latency is below threshold
        if stats['latency_current_ms'] > self.config.realtime.backpressure_threshold_ms:
            return False
        
        # Check drop rate is reasonable (< 10%)
        if stats['drop_rate_percent'] > 10.0:
            return False
        
        return True
    
    def _report_detection_events(self):
        """Report detection events to the callback."""
        try:
            # Get active tracks with matches
            active_tracks = self.pipeline.tracker.get_active_tracks()
            
            for track in active_tracks:
                if track.matches:
                    best_match = track.get_best_match()
                    if best_match:
                        # Report the detection event
                        self.event_callback(
                            category=best_match.category,
                            masked_text=best_match.masked_text,
                            confidence=best_match.confidence
                        )
        except Exception as e:
            logger.warning(f"Error reporting detection events: {e}")