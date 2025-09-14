"""Optimized real-time privacy detection with smooth performance."""

import logging
import time
import threading
from typing import Optional, Dict, Any, List
import numpy as np
import cv2
from queue import Queue, Empty
from collections import deque

from .optimized_detector import OptimizedPrivacyDetector, OptimizedDetection

logger = logging.getLogger(__name__)


class OptimizedRealtimeDetector:
    """Optimized real-time privacy detector with smooth performance."""
    
    def __init__(self):
        """Initialize optimized real-time detector."""
        self.detector = OptimizedPrivacyDetector()
        
        # Threading components
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_queue = Queue(maxsize=2)  # Small queue for low latency
        
        # Current state
        self._current_frame: Optional[np.ndarray] = None
        self._current_detections: List[OptimizedDetection] = []
        self._frame_lock = threading.Lock()
        
        # Performance tracking
        self._fps_history = deque(maxlen=30)
        self._processing_times = deque(maxlen=30)
        self._current_fps = 0.0
        self._target_fps = 60
        
        # Adaptive settings
        self.adaptive_quality = True
        self.current_quality_scale = 1.0
        self.min_quality_scale = 0.6
        self.max_quality_scale = 1.0
        
        # Frame skipping
        self.frame_skip_enabled = True
        self.frame_skip_count = 0
        self.max_frame_skip = 1  # Conservative skipping
        
        # Display settings
        self.show_performance = True
        self.show_primary_secondary = True
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'quality_adjustments': 0,
            'primary_faces_detected': 0,
            'secondary_faces_detected': 0
        }
        
        logger.info("Optimized real-time detector initialized")
    
    def start(self):
        """Start optimized real-time detection."""
        if self._processing_thread and self._processing_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        
        logger.info("Optimized real-time detector started")
    
    def stop(self):
        """Stop optimized real-time detection."""
        if not self._processing_thread:
            return
        
        self._stop_event.set()
        
        # Clear queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break
        
        if self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.0)
        
        self._processing_thread = None
        self.detector.cleanup()
        logger.info("Optimized real-time detector stopped")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with optimized detection."""
        if frame is None or frame.size == 0:
            return frame
        
        frame_start_time = time.time()
        
        # Store current frame
        with self._frame_lock:
            self._current_frame = frame.copy()
        
        # Adaptive frame skipping
        if self._should_skip_frame():
            self.stats['frames_skipped'] += 1
            return self._draw_cached_results(frame)
        
        # Adaptive quality scaling
        processed_frame = self._apply_quality_scaling(frame)
        
        # Add to processing queue (non-blocking)
        try:
            # Clear old frames from queue to maintain low latency
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except Empty:
                    break
            
            self._frame_queue.put_nowait(processed_frame)
        except:
            pass  # Queue full, continue with cached results
        
        # Draw current detections
        result_frame = self._draw_cached_results(frame)
        
        # Update FPS tracking
        frame_time = time.time() - frame_start_time
        self._update_fps_tracking(frame_time)
        
        # Adaptive performance adjustment
        if self.adaptive_quality:
            self._adjust_performance_settings()
        
        return result_frame
    
    def _processing_loop(self):
        """Optimized processing loop."""
        logger.info("Optimized processing loop started")
        
        while not self._stop_event.is_set():
            try:
                # Get frame with timeout
                try:
                    frame = self._frame_queue.get(timeout=0.033)  # 30 FPS timeout
                except Empty:
                    continue
                
                # Process frame
                processing_start = time.time()
                detections = self.detector.detect_privacy_violations(frame)
                processing_time = time.time() - processing_start
                
                # Update current detections
                with self._frame_lock:
                    self._current_detections = detections
                
                # Update statistics
                self.stats['frames_processed'] += 1
                self._processing_times.append(processing_time)
                
                # Count primary/secondary faces
                for detection in detections:
                    if detection.category == 'face':
                        if detection.priority == 'primary':
                            self.stats['primary_faces_detected'] += 1
                        elif detection.priority == 'secondary':
                            self.stats['secondary_faces_detected'] += 1
                
                self._frame_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in optimized processing loop: {e}")
                continue
        
        logger.info("Optimized processing loop finished")
    
    def _should_skip_frame(self) -> bool:
        """Determine if frame should be skipped."""
        if not self.frame_skip_enabled:
            return False
        
        current_fps = self._get_current_fps()
        if current_fps < self._target_fps * 0.85:  # If below 85% of target
            self.frame_skip_count += 1
            if self.frame_skip_count <= self.max_frame_skip:
                return True
        
        self.frame_skip_count = 0
        return False
    
    def _apply_quality_scaling(self, frame: np.ndarray) -> np.ndarray:
        """Apply adaptive quality scaling."""
        if self.current_quality_scale == 1.0:
            return frame
        
        new_width = int(frame.shape[1] * self.current_quality_scale)
        new_height = int(frame.shape[0] * self.current_quality_scale)
        
        return cv2.resize(frame, (new_width, new_height))
    
    def _draw_cached_results(self, frame: np.ndarray) -> np.ndarray:
        """Draw cached detection results."""
        result = frame.copy()
        
        with self._frame_lock:
            current_detections = self._current_detections.copy()
        
        if current_detections:
            result = self.detector.draw_detections(result, current_detections)
        
        if self.show_performance:
            self._draw_performance_overlay(result)
        
        return result
    
    def _draw_performance_overlay(self, frame: np.ndarray):
        """Draw performance overlay."""
        stats = self.get_comprehensive_stats()
        
        # Performance info
        perf_lines = [
            f"FPS: {stats['current_fps']:.1f} / {self._target_fps}",
            f"Quality: {self.current_quality_scale:.2f}",
            f"Processing: {stats['avg_processing_ms']:.1f}ms"
        ]
        
        # Detection info
        primary_faces = sum(1 for d in self._current_detections 
                          if d.category == 'face' and d.priority == 'primary')
        secondary_faces = sum(1 for d in self._current_detections 
                            if d.category == 'face' and d.priority == 'secondary')
        other_detections = len([d for d in self._current_detections if d.category != 'face'])
        
        if primary_faces > 0 or secondary_faces > 0:
            perf_lines.append(f"Primary Faces: {primary_faces}")
            perf_lines.append(f"Secondary Faces: {secondary_faces}")
        
        if other_detections > 0:
            perf_lines.append(f"Other: {other_detections}")
        
        # Performance status
        performance_ratio = stats['current_fps'] / self._target_fps
        if performance_ratio >= 0.9:
            status_color = (0, 255, 0)  # Green
            status = "SMOOTH"
        elif performance_ratio >= 0.7:
            status_color = (0, 255, 255)  # Yellow
            status = "GOOD"
        else:
            status_color = (0, 0, 255)  # Red
            status = "CHOPPY"
        
        perf_lines.insert(0, f"Status: {status}")
        
        # Draw background
        overlay_height = len(perf_lines) * 25 + 15
        cv2.rectangle(frame, (10, 10), (280, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (280, overlay_height), status_color, 2)
        
        # Draw text
        for i, line in enumerate(perf_lines):
            y_pos = 35 + i * 25
            color = status_color if i == 0 else (255, 255, 255)
            cv2.putText(frame, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _update_fps_tracking(self, frame_time: float):
        """Update FPS tracking."""
        self._fps_history.append(frame_time)
        
        if len(self._fps_history) > 0:
            avg_frame_time = sum(self._fps_history) / len(self._fps_history)
            self._current_fps = 1.0 / max(avg_frame_time, 0.001)
    
    def _get_current_fps(self) -> float:
        """Get current FPS."""
        return self._current_fps
    
    def _adjust_performance_settings(self):
        """Adjust performance settings adaptively."""
        current_fps = self._get_current_fps()
        target_fps = self._target_fps
        
        # Adjust quality scale
        if current_fps < target_fps * 0.8:  # Below 80% of target
            new_scale = max(self.current_quality_scale * 0.95, self.min_quality_scale)
            if new_scale != self.current_quality_scale:
                self.current_quality_scale = new_scale
                self.stats['quality_adjustments'] += 1
                logger.debug(f"Decreased quality to {new_scale:.2f}")
        
        elif current_fps > target_fps * 1.1:  # Above 110% of target
            new_scale = min(self.current_quality_scale * 1.02, self.max_quality_scale)
            if new_scale != self.current_quality_scale:
                self.current_quality_scale = new_scale
                self.stats['quality_adjustments'] += 1
                logger.debug(f"Increased quality to {new_scale:.2f}")
    
    def get_current_detections(self) -> List[OptimizedDetection]:
        """Get current detections."""
        with self._frame_lock:
            return self._current_detections.copy()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        detector_stats = self.detector.get_stats()
        
        avg_processing_time = (
            sum(self._processing_times) / max(len(self._processing_times), 1)
            if self._processing_times else 0.0
        )
        
        return {
            **self.stats,
            **detector_stats,
            'current_fps': self._current_fps,
            'target_fps': self._target_fps,
            'quality_scale': self.current_quality_scale,
            'avg_processing_ms': avg_processing_time * 1000,
            'performance_ratio': self._current_fps / max(self._target_fps, 1),
            'active_detections': len(self._current_detections)
        }
    
    def set_target_fps(self, fps: int):
        """Set target FPS."""
        self._target_fps = fps
        self.detector.set_target_fps(fps)
        logger.info(f"Target FPS set to {fps}")
    
    def set_adaptive_quality(self, enabled: bool):
        """Enable/disable adaptive quality."""
        self.adaptive_quality = enabled
        if not enabled:
            self.current_quality_scale = 1.0
        logger.info(f"Adaptive quality {'enabled' if enabled else 'disabled'}")
    
    def set_frame_skipping(self, enabled: bool):
        """Enable/disable frame skipping."""
        self.frame_skip_enabled = enabled
        logger.info(f"Frame skipping {'enabled' if enabled else 'disabled'}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()