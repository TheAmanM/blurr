"""Real-time privacy violation detection for video streams."""

import logging
import time
import threading
from typing import Optional, Callable, Dict, Any
import numpy as np
import cv2
from queue import Queue, Empty

from .onnx_detector import ONNXPrivacyDetector, PrivacyDetection

logger = logging.getLogger(__name__)


class RealtimePrivacyDetector:
    """Real-time privacy violation detector for video streams."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize real-time detector.
        
        Args:
            model_path: Path to ONNX model file
        """
        self.detector = ONNXPrivacyDetector(model_path)
        
        # Threading components
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_queue = Queue(maxsize=2)  # Small queue to avoid lag
        self._result_queue = Queue(maxsize=10)
        
        # Current state
        self._current_frame: Optional[np.ndarray] = None
        self._current_detections: list = []
        self._frame_lock = threading.Lock()
        
        # Performance tracking
        self._fps_counter = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0
        
        # Configuration
        self.draw_boxes = True
        self.show_labels = True
        self.show_confidence = True
        
    def start(self):
        """Start the real-time detection thread."""
        if self._processing_thread and self._processing_thread.is_alive():
            logger.warning("Real-time detector already running")
            return
        
        self._stop_event.clear()
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        logger.info("Real-time privacy detector started")
    
    def stop(self):
        """Stop the real-time detection thread."""
        if not self._processing_thread:
            return
        
        logger.info("Stopping real-time privacy detector...")
        self._stop_event.set()
        
        # Clear queues
        self._clear_queues()
        
        # Wait for thread to finish
        if self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)
            if self._processing_thread.is_alive():
                logger.warning("Processing thread did not stop gracefully")
        
        self._processing_thread = None
        self.detector.cleanup()
        logger.info("Real-time privacy detector stopped")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame and return result with detections drawn.
        
        Args:
            frame: Input frame as BGR numpy array
            
        Returns:
            Frame with privacy violations highlighted
        """
        if frame is None or frame.size == 0:
            return frame
        
        # Store current frame for processing thread
        with self._frame_lock:
            self._current_frame = frame.copy()
        
        # Add frame to processing queue (non-blocking)
        try:
            self._frame_queue.put_nowait(frame.copy())
        except:
            # Queue full, skip this frame to maintain real-time performance
            pass
        
        # Get latest detections and draw them
        result_frame = frame.copy()
        
        if self.draw_boxes and self._current_detections:
            result_frame = self.detector.draw_detections(result_frame, self._current_detections)
        
        # Update FPS counter
        self._update_fps()
        
        # Draw FPS and stats
        if self.show_labels:
            self._draw_stats(result_frame)
        
        return result_frame
    
    def _processing_loop(self):
        """Main processing loop running in separate thread."""
        logger.info("Privacy detection processing loop started")
        
        while not self._stop_event.is_set():
            try:
                # Get frame from queue with timeout
                try:
                    frame = self._frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Process frame
                detections = self.detector.detect_privacy_violations(frame)
                
                # Update current detections
                with self._frame_lock:
                    self._current_detections = detections
                
                # Mark task as done
                self._frame_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                continue
        
        logger.info("Privacy detection processing loop finished")
    
    def _clear_queues(self):
        """Clear all queues."""
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break
        
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except Empty:
                break
    
    def _update_fps(self):
        """Update FPS counter."""
        self._fps_counter += 1
        current_time = time.time()
        
        if current_time - self._fps_start_time >= 1.0:
            self._current_fps = self._fps_counter / (current_time - self._fps_start_time)
            self._fps_counter = 0
            self._fps_start_time = current_time
    
    def _draw_stats(self, frame: np.ndarray):
        """Draw performance statistics on frame."""
        stats = self.get_stats()
        
        # Prepare stats text
        stats_lines = [
            f"FPS: {self._current_fps:.1f}",
            f"Detections: {len(self._current_detections)}",
            f"Provider: {stats.get('provider', 'Unknown')}",
            f"Inference: {stats.get('avg_inference_time', 0)*1000:.1f}ms"
        ]
        
        # Draw background
        y_offset = 30
        for i, line in enumerate(stats_lines):
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                frame,
                (10, y_offset + i * 25 - 20),
                (20 + text_size[0], y_offset + i * 25 + 5),
                (0, 0, 0),
                -1
            )
        
        # Draw text
        for i, line in enumerate(stats_lines):
            cv2.putText(
                frame,
                line,
                (15, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
    
    def get_current_detections(self) -> list:
        """Get current privacy detections."""
        with self._frame_lock:
            return self._current_detections.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        detector_stats = self.detector.get_stats()
        
        return {
            **detector_stats,
            'current_fps': self._current_fps,
            'queue_size': self._frame_queue.qsize(),
            'active_detections': len(self._current_detections)
        }
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for detections."""
        self.detector.set_confidence_threshold(threshold)
    
    def set_draw_boxes(self, draw: bool):
        """Enable/disable drawing bounding boxes."""
        self.draw_boxes = draw
    
    def set_show_labels(self, show: bool):
        """Enable/disable showing labels and stats."""
        self.show_labels = show
    
    def set_show_confidence(self, show: bool):
        """Enable/disable showing confidence scores."""
        self.show_confidence = show
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class PrivacyViolationCounter:
    """Utility class to count and track privacy violations over time."""
    
    def __init__(self, window_size: int = 30):
        """Initialize violation counter.
        
        Args:
            window_size: Number of frames to consider for statistics
        """
        self.window_size = window_size
        self.violation_history = []
        self.category_counts = {}
        
    def update(self, detections: list):
        """Update with new detections."""
        # Add current frame detections
        frame_violations = {
            'timestamp': time.time(),
            'count': len(detections),
            'categories': {}
        }
        
        # Count by category
        for detection in detections:
            category = detection.category
            frame_violations['categories'][category] = (
                frame_violations['categories'].get(category, 0) + 1
            )
        
        self.violation_history.append(frame_violations)
        
        # Keep only recent history
        if len(self.violation_history) > self.window_size:
            self.violation_history.pop(0)
        
        # Update overall category counts
        self._update_category_counts()
    
    def _update_category_counts(self):
        """Update category counts from recent history."""
        self.category_counts = {}
        
        for frame in self.violation_history:
            for category, count in frame['categories'].items():
                self.category_counts[category] = (
                    self.category_counts.get(category, 0) + count
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get violation statistics."""
        if not self.violation_history:
            return {
                'total_violations': 0,
                'avg_violations_per_frame': 0.0,
                'category_counts': {},
                'recent_trend': 'stable'
            }
        
        total_violations = sum(frame['count'] for frame in self.violation_history)
        avg_violations = total_violations / len(self.violation_history)
        
        # Calculate trend
        if len(self.violation_history) >= 10:
            recent_avg = sum(frame['count'] for frame in self.violation_history[-5:]) / 5
            older_avg = sum(frame['count'] for frame in self.violation_history[-10:-5]) / 5
            
            if recent_avg > older_avg * 1.2:
                trend = 'increasing'
            elif recent_avg < older_avg * 0.8:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'total_violations': total_violations,
            'avg_violations_per_frame': avg_violations,
            'category_counts': self.category_counts.copy(),
            'recent_trend': trend,
            'frames_analyzed': len(self.violation_history)
        }