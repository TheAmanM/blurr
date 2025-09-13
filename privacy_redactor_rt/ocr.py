"""Asynchronous OCR worker with queue management and text normalization."""

import logging
import queue
import threading
import time
import unicodedata
from typing import Dict, Optional, Tuple
import numpy as np

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PaddleOCR = None
    PADDLEOCR_AVAILABLE = False

from .config import OCRConfig
from .types import BBox

logger = logging.getLogger(__name__)


class OCRWorker:
    """Asynchronous OCR worker with bounded queue and result caching."""
    
    def __init__(self, config: OCRConfig, max_queue_size: int = 2):
        """Initialize OCR worker with configuration.
        
        Args:
            config: OCR configuration settings
            max_queue_size: Maximum number of items in processing queue
        """
        self.config = config
        self.max_queue_size = max_queue_size
        
        # Threading components
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        
        # Result cache with thread-safe access
        self._results_lock = threading.Lock()
        self._results: Dict[str, Tuple[str, float]] = {}  # track_id -> (text, timestamp)
        
        # OCR engine (lazy initialization)
        self._ocr_engine: Optional[PaddleOCR] = None
        self._ocr_lock = threading.Lock()
        
        # Statistics
        self._stats_lock = threading.Lock()
        self._processed_count = 0
        self._queue_full_count = 0
        self._cache_hits = 0
        
    def _lazy_init_ocr(self):
        """Initialize OCR engine with lazy loading."""
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR is not available. Please install paddleocr package.")
            
        if self._ocr_engine is None:
            with self._ocr_lock:
                if self._ocr_engine is None:
                    logger.info("Initializing PaddleOCR engine...")
                    self._ocr_engine = PaddleOCR(
                        lang='en',
                        ocr_version='PP-OCRv4'
                    )
                    logger.info("PaddleOCR engine initialized")
        return self._ocr_engine
    
    def start(self) -> None:
        """Start the OCR worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.warning("OCR worker already running")
            return
            
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("OCR worker started")
    
    def stop(self) -> None:
        """Stop the OCR worker thread and cleanup."""
        if self._worker_thread is None:
            return
            
        logger.info("Stopping OCR worker...")
        self._stop_event.set()
        
        # Clear the queue to unblock any waiting operations
        try:
            while not self._queue.empty():
                self._queue.get_nowait()
        except queue.Empty:
            pass
            
        # Wait for worker thread to finish
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                logger.warning("OCR worker thread did not stop gracefully")
        
        self._worker_thread = None
        logger.info("OCR worker stopped")
    
    def enqueue_roi(self, roi: np.ndarray, track_id: str, bbox: BBox) -> bool:
        """Enqueue ROI for OCR processing.
        
        Args:
            roi: Region of interest image array (BGR format)
            track_id: Unique identifier for tracking
            bbox: Bounding box information
            
        Returns:
            True if successfully enqueued, False if queue is full
        """
        if roi is None or roi.size == 0:
            logger.warning(f"Invalid ROI for track {track_id}")
            return False
            
        # Validate ROI dimensions
        if len(roi.shape) != 3 or roi.shape[2] != 3:
            logger.warning(f"Invalid ROI shape {roi.shape} for track {track_id}")
            return False
            
        try:
            # Non-blocking enqueue with immediate return if full
            self._queue.put_nowait({
                'roi': roi.copy(),  # Copy to avoid race conditions
                'track_id': track_id,
                'bbox': bbox,
                'timestamp': time.time()
            })
            return True
            
        except queue.Full:
            with self._stats_lock:
                self._queue_full_count += 1
            logger.debug(f"OCR queue full, dropping ROI for track {track_id}")
            return False
    
    def get_result(self, track_id: str) -> Optional[str]:
        """Get cached OCR result for track ID.
        
        Args:
            track_id: Track identifier
            
        Returns:
            OCR text result or None if not available
        """
        with self._results_lock:
            if track_id in self._results:
                text, timestamp = self._results[track_id]
                with self._stats_lock:
                    self._cache_hits += 1
                return text
        return None
    
    def clear_result(self, track_id: str) -> None:
        """Clear cached result for track ID."""
        with self._results_lock:
            self._results.pop(track_id, None)
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        with self._stats_lock:
            return {
                'processed_count': self._processed_count,
                'queue_full_count': self._queue_full_count,
                'cache_hits': self._cache_hits,
                'queue_size': self._queue.qsize(),
                'cache_size': len(self._results)
            }
    
    def _worker_loop(self) -> None:
        """Main worker thread loop for processing OCR requests."""
        logger.info("OCR worker loop started")
        
        while not self._stop_event.is_set():
            try:
                # Get next item from queue with timeout
                try:
                    item = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the OCR request
                self._process_ocr_request(item)
                self._queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in OCR worker loop: {e}")
                continue
        
        logger.info("OCR worker loop finished")
    
    def _process_ocr_request(self, item: Dict) -> None:
        """Process a single OCR request.
        
        Args:
            item: Dictionary containing roi, track_id, bbox, and timestamp
        """
        roi = item['roi']
        track_id = item['track_id']
        bbox = item['bbox']
        timestamp = item['timestamp']
        
        try:
            # Initialize OCR engine if needed
            ocr_engine = self._lazy_init_ocr()
            
            # Perform OCR on the ROI
            start_time = time.time()
            results = ocr_engine.predict(roi)
            ocr_time = time.time() - start_time
            
            # Extract and normalize text
            text = self._extract_and_normalize_text(results)
            
            # Cache the result
            with self._results_lock:
                self._results[track_id] = (text, timestamp)
                
                # Cleanup old results (keep last 100)
                if len(self._results) > 100:
                    oldest_key = min(self._results.keys(), 
                                   key=lambda k: self._results[k][1])
                    del self._results[oldest_key]
            
            with self._stats_lock:
                self._processed_count += 1
            
            logger.debug(f"OCR completed for track {track_id} in {ocr_time:.3f}s: '{text[:50]}...'")
            
        except Exception as e:
            logger.error(f"OCR processing failed for track {track_id}: {e}")
            # Cache empty result to avoid repeated processing
            with self._results_lock:
                self._results[track_id] = ("", timestamp)
    
    def _extract_and_normalize_text(self, ocr_results) -> str:
        """Extract and normalize text from OCR results.
        
        Args:
            ocr_results: Raw OCR results from PaddleOCR predict method
            
        Returns:
            Normalized text string
        """
        if not ocr_results or len(ocr_results) == 0:
            return ""
        
        # Extract text from new OCR results format
        text_parts = []
        result = ocr_results[0]  # Get first result
        
        rec_texts = result.get('rec_texts', [])
        rec_scores = result.get('rec_scores', [])
        
        for i, text_content in enumerate(rec_texts):
            confidence = rec_scores[i] if i < len(rec_scores) else 1.0
            
            # Filter by confidence threshold
            if confidence >= self.config.min_ocr_confidence:
                text_parts.append(str(text_content))
        
        # Join all text parts
        raw_text = " ".join(text_parts)
        
        # Apply text normalization
        normalized_text = self._normalize_text(raw_text)
        
        # Truncate if too long
        if len(normalized_text) > self.config.max_text_length:
            normalized_text = normalized_text[:self.config.max_text_length]
        
        return normalized_text
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text using NFKC and whitespace handling.
        
        Args:
            text: Raw text string
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
        
        # Unicode normalization (NFKC - compatibility decomposition + canonical composition)
        normalized = unicodedata.normalize('NFKC', text)
        
        # Whitespace normalization
        # Replace multiple whitespace characters with single space
        import re
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Strip leading/trailing whitespace
        normalized = normalized.strip()
        
        # Preserve case - no case conversion applied
        
        return normalized
    
    def cleanup_expired_results(self, max_age_seconds: float = 300.0) -> int:
        """Clean up expired cached results.
        
        Args:
            max_age_seconds: Maximum age for cached results
            
        Returns:
            Number of results cleaned up
        """
        current_time = time.time()
        expired_keys = []
        
        with self._results_lock:
            for track_id, (text, timestamp) in self._results.items():
                if current_time - timestamp > max_age_seconds:
                    expired_keys.append(track_id)
            
            for key in expired_keys:
                del self._results[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired OCR results")
        
        return len(expired_keys)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()