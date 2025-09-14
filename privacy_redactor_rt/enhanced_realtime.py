"""Real-time enhanced privacy detection with optimized performance."""

import logging
import time
import threading
from typing import Optional, Dict, Any, List
import numpy as np
import cv2
from queue import Queue, Empty
from collections import deque

from .enhanced_detector import EnhancedPrivacyDetector, EnhancedDetection

logger = logging.getLogger(__name__)


class EnhancedRealtimeDetector:
    """Real-time enhanced privacy detector with advanced optimizations."""
    
    def __init__(self):
        """Initialize enhanced real-time detector."""
        self.detector = EnhancedPrivacyDetector()
        
        # Threading components
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_queue = Queue(maxsize=1)  # Very small queue for minimal latency
        
        # Current state
        self._current_frame: Optional[np.ndarray] = None
        self._current_detections: List[EnhancedDetection] = []
        self._frame_lock = threading.Lock()
        
        # Performance optimization
        self._fps_history = deque(maxlen=30)  # Track FPS over 30 frames
        self._processing_times = deque(maxlen=30)
        self._current_fps = 0.0
        self._target_fps = 60
        
        # Adaptive quality settings
        self.adaptive_quality = True
        self.current_quality_scale = 1.0
        self.min_quality_scale = 0.5
        self.max_quality_scale = 1.0
        
        # Frame skipping for performance
        self.frame_skip_enabled = True
        self.frame_skip_count = 0
        self.max_frame_skip = 2
        
        # Detection settings
        self.confidence_threshold = 0.7
        self.draw_boxes = True
        self.show_labels = True
        self.show_performance = True
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'detections_total': 0,
            'avg_detections_per_frame': 0.0,
            'quality_adjustments': 0
        }
        
        logger.info("Enhanced real-time detector initialized")
    
    def start(self):
        """Start the enhanced real-time detection."""
        if self._processing_thread and self._processing_thread.is_alive():
            logger.warning("Enhanced detector already running")
            return
        
        self._stop_event.clear()
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        
        # Set initial target FPS
        self.detector.set_target_fps(self._target_fps)
        self.detector.set_adaptive_processing(True)
        
        logger.info("Enhanced real-time detector started")
    
    def stop(self):
        """Stop the enhanced real-time detection."""
        if not self._processing_thread:
            return
        
        logger.info("Stopping enhanced real-time detector...")
        self._stop_event.set()
        
        # Clear queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break
        
        # Wait for thread
        if self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)
        
        self._processing_thread = None
        self.detector.cleanup()
        logger.info("Enhanced real-time detector stopped")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with enhanced detection and optimizations."""
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
            self._frame_queue.put_nowait(processed_frame)
        except:
            # Queue full, continue with cached results
            pass
        
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
        """Enhanced processing loop with optimizations."""
        logger.info("Enhanced processing loop started")
        
        while not self._stop_event.is_set():
            try:
                # Get frame with timeout
                try:
                    frame = self._frame_queue.get(timeout=0.05)
                except Empty:
                    continue
                
                # Process frame
                processing_start = time.time()
                detections = self.detector.detect_privacy_violations(frame)
                processing_time = time.time() - processing_start
                
                # Filter by confidence threshold
                filtered_detections = [
                    d for d in detections if d.confidence >= self.confidence_threshold
                ]
                
                # Update current detections
                with self._frame_lock:
                    self._current_detections = filtered_detections
                
                # Update statistics
                self.stats['frames_processed'] += 1
                self.stats['detections_total'] += len(filtered_detections)
                self.stats['avg_detections_per_frame'] = (
                    self.stats['detections_total'] / max(self.stats['frames_processed'], 1)
                )
                
                # Track processing times
                self._processing_times.append(processing_time)
                
                self._frame_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in enhanced processing loop: {e}")
                continue
        
        logger.info("Enhanced processing loop finished")
    
    def _should_skip_frame(self) -> bool:
        """Determine if current frame should be skipped for performance."""
        if not self.frame_skip_enabled:
            return False
        
        # Skip frames if processing is falling behind
        current_fps = self._get_current_fps()
        if current_fps < self._target_fps * 0.8:  # If below 80% of target
            self.frame_skip_count += 1
            if self.frame_skip_count <= self.max_frame_skip:
                return True
        
        self.frame_skip_count = 0
        return False
    
    def _apply_quality_scaling(self, frame: np.ndarray) -> np.ndarray:
        """Apply adaptive quality scaling for performance."""
        if self.current_quality_scale == 1.0:
            return frame
        
        # Scale down for processing
        new_width = int(frame.shape[1] * self.current_quality_scale)
        new_height = int(frame.shape[0] * self.current_quality_scale)
        
        scaled_frame = cv2.resize(frame, (new_width, new_height))
        return scaled_frame
    
    def _draw_cached_results(self, frame: np.ndarray) -> np.ndarray:
        """Draw cached detection results on frame."""
        result = frame.copy()
        
        with self._frame_lock:
            current_detections = self._current_detections.copy()
        
        if self.draw_boxes and current_detections:
            result = self.detector.draw_enhanced_detections(result, current_detections)
        
        if self.show_performance:
            self._draw_performance_overlay(result)
        
        return result
    
    def _draw_performance_overlay(self, frame: np.ndarray):
        """Draw performance information overlay."""
        stats = self.get_comprehensive_stats()
        
        # Performance info
        perf_lines = [
            f"FPS: {stats['current_fps']:.1f} / {self._target_fps}",
            f"Detections: {len(self._current_detections)}",
            f"Quality: {self.current_quality_scale:.2f}",
            f"Processing: {stats['avg_processing_ms']:.1f}ms",
            f"Efficiency: {stats['efficiency']:.1f}x"
        ]
        
        # Category breakdown
        category_counts = {}
        for detection in self._current_detections:
            category_counts[detection.category] = category_counts.get(detection.category, 0) + 1
        
        if category_counts:
            perf_lines.append("Categories:")
            for category, count in category_counts.items():
                perf_lines.append(f"  {category}: {count}")
        
        # Draw background
        overlay_height = len(perf_lines) * 20 + 10
        cv2.rectangle(frame, (10, 10), (300, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, overlay_height), (0, 255, 0), 2)
        
        # Draw text
        for i, line in enumerate(perf_lines):
            y_pos = 30 + i * 20
            color = (0, 255, 0) if not line.startswith("  ") else (200, 200, 200)
            cv2.putText(frame, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _update_fps_tracking(self, frame_time: float):
        """Update FPS tracking with current frame time."""
        self._fps_history.append(frame_time)
        
        if len(self._fps_history) > 0:
            avg_frame_time = sum(self._fps_history) / len(self._fps_history)
            self._current_fps = 1.0 / max(avg_frame_time, 0.001)
    
    def _get_current_fps(self) -> float:
        """Get current FPS."""
        return self._current_fps
    
    def _adjust_performance_settings(self):
        """Adjust performance settings based on current performance."""
        current_fps = self._get_current_fps()
        target_fps = self._target_fps
        
        # Adjust quality scale based on performance
        if current_fps < target_fps * 0.7:  # Below 70% of target
            # Decrease quality for better performance
            new_scale = max(self.current_quality_scale * 0.9, self.min_quality_scale)
            if new_scale != self.current_quality_scale:
                self.current_quality_scale = new_scale
                self.stats['quality_adjustments'] += 1
                logger.debug(f"Decreased quality scale to {new_scale:.2f}")
        
        elif current_fps > target_fps * 1.2:  # Above 120% of target
            # Increase quality if we have headroom
            new_scale = min(self.current_quality_scale * 1.05, self.max_quality_scale)
            if new_scale != self.current_quality_scale:
                self.current_quality_scale = new_scale
                self.stats['quality_adjustments'] += 1
                logger.debug(f"Increased quality scale to {new_scale:.2f}")
    
    def get_current_detections(self) -> List[EnhancedDetection]:
        """Get current enhanced detections."""
        with self._frame_lock:
            return self._current_detections.copy()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        detector_stats = self.detector.get_stats()
        
        # Calculate processing time statistics
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
            'frame_skip_enabled': self.frame_skip_enabled,
            'adaptive_quality': self.adaptive_quality,
            'queue_size': self._frame_queue.qsize(),
            'active_detections': len(self._current_detections),
            'performance_ratio': self._current_fps / max(self._target_fps, 1)
        }
    
    def set_target_fps(self, fps: int):
        """Set target FPS."""
        self._target_fps = fps
        self.detector.set_target_fps(fps)
        logger.info(f"Target FPS set to {fps}")
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for detections."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold set to {self.confidence_threshold}")
    
    def set_adaptive_quality(self, enabled: bool):
        """Enable/disable adaptive quality scaling."""
        self.adaptive_quality = enabled
        if not enabled:
            self.current_quality_scale = 1.0
        logger.info(f"Adaptive quality {'enabled' if enabled else 'disabled'}")
    
    def set_frame_skipping(self, enabled: bool):
        """Enable/disable frame skipping optimization."""
        self.frame_skip_enabled = enabled
        logger.info(f"Frame skipping {'enabled' if enabled else 'disabled'}")
    
    def set_draw_boxes(self, draw: bool):
        """Enable/disable drawing bounding boxes."""
        self.draw_boxes = draw
    
    def set_show_labels(self, show: bool):
        """Enable/disable showing labels."""
        self.show_labels = show
    
    def set_show_performance(self, show: bool):
        """Enable/disable performance overlay."""
        self.show_performance = show
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'detections_total': 0,
            'avg_detections_per_frame': 0.0,
            'quality_adjustments': 0
        }
        self._fps_history.clear()
        self._processing_times.clear()
        logger.info("Statistics reset")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class EnhancedViolationAnalyzer:
    """Advanced analyzer for privacy violation patterns and trends."""
    
    def __init__(self, history_size: int = 100):
        """Initialize violation analyzer."""
        self.history_size = history_size
        self.detection_history = deque(maxlen=history_size)
        self.category_trends = {}
        self.confidence_trends = {}
        
    def update(self, detections: List[EnhancedDetection]):
        """Update analyzer with new detections."""
        timestamp = time.time()
        
        # Store detection snapshot
        snapshot = {
            'timestamp': timestamp,
            'detections': detections.copy(),
            'total_count': len(detections),
            'categories': {},
            'avg_confidence': 0.0
        }
        
        # Analyze categories and confidence
        if detections:
            category_counts = {}
            total_confidence = 0.0
            
            for detection in detections:
                category = detection.category
                category_counts[category] = category_counts.get(category, 0) + 1
                total_confidence += detection.confidence
            
            snapshot['categories'] = category_counts
            snapshot['avg_confidence'] = total_confidence / len(detections)
        
        self.detection_history.append(snapshot)
        self._update_trends()
    
    def _update_trends(self):
        """Update trend analysis."""
        if len(self.detection_history) < 10:
            return
        
        # Analyze category trends over recent history
        recent_snapshots = list(self.detection_history)[-20:]  # Last 20 frames
        
        for category in ['face', 'license_plate', 'document', 'screen']:
            counts = [
                snapshot['categories'].get(category, 0)
                for snapshot in recent_snapshots
            ]
            
            if len(counts) >= 10:
                # Simple trend calculation
                first_half = sum(counts[:len(counts)//2])
                second_half = sum(counts[len(counts)//2:])
                
                if first_half == 0 and second_half == 0:
                    trend = 'stable'
                elif second_half > first_half * 1.5:
                    trend = 'increasing'
                elif second_half < first_half * 0.5:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
                
                self.category_trends[category] = trend
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get comprehensive violation analysis."""
        if not self.detection_history:
            return {'status': 'no_data'}
        
        recent_snapshots = list(self.detection_history)[-30:]  # Last 30 frames
        
        # Overall statistics
        total_detections = sum(s['total_count'] for s in recent_snapshots)
        avg_detections_per_frame = total_detections / len(recent_snapshots)
        
        # Category breakdown
        category_totals = {}
        confidence_by_category = {}
        
        for snapshot in recent_snapshots:
            for category, count in snapshot['categories'].items():
                category_totals[category] = category_totals.get(category, 0) + count
                
                # Track confidence by category
                if category not in confidence_by_category:
                    confidence_by_category[category] = []
                
                # Get confidence for this category in this frame
                category_detections = [
                    d for d in snapshot['detections'] if d.category == category
                ]
                if category_detections:
                    avg_conf = sum(d.confidence for d in category_detections) / len(category_detections)
                    confidence_by_category[category].append(avg_conf)
        
        # Calculate average confidence by category
        avg_confidence_by_category = {}
        for category, confidences in confidence_by_category.items():
            if confidences:
                avg_confidence_by_category[category] = sum(confidences) / len(confidences)
        
        return {
            'status': 'active',
            'frames_analyzed': len(recent_snapshots),
            'total_detections': total_detections,
            'avg_detections_per_frame': avg_detections_per_frame,
            'category_totals': category_totals,
            'category_trends': self.category_trends.copy(),
            'avg_confidence_by_category': avg_confidence_by_category,
            'most_common_category': max(category_totals.items(), key=lambda x: x[1])[0] if category_totals else None,
            'detection_rate': len([s for s in recent_snapshots if s['total_count'] > 0]) / len(recent_snapshots)
        }