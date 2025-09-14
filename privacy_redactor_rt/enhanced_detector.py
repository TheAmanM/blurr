"""Enhanced privacy detection with improved accuracy and FPS optimization."""

import logging
import time
import threading
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import cv2
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False

from .types import BBox

logger = logging.getLogger(__name__)


@dataclass
class EnhancedDetection:
    """Enhanced privacy detection result with confidence scoring."""
    bbox: BBox
    category: str
    confidence: float
    sub_category: Optional[str] = None  # e.g., 'frontal_face', 'profile_face'
    features: Optional[Dict[str, float]] = None  # Additional classification features
    label: Optional[str] = None


class SpecializedClassifier:
    """Specialized classifier for specific privacy categories."""
    
    def __init__(self, category: str):
        self.category = category
        self.confidence_threshold = 0.7
        self._initialize_category_specific()
    
    def _initialize_category_specific(self):
        """Initialize category-specific classifiers."""
        if self.category == 'face':
            self._init_face_classifier()
        elif self.category == 'license_plate':
            self._init_license_plate_classifier()
        elif self.category == 'document':
            self._init_document_classifier()
        elif self.category == 'screen':
            self._init_screen_classifier()
    
    def _init_face_classifier(self):
        """Initialize face-specific classification."""
        # Load multiple face detection models for better accuracy
        self.face_cascade_frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_cascade_profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        
        # Face quality thresholds
        self.min_face_size = (40, 40)
        self.max_face_size = (300, 300)
        self.aspect_ratio_range = (0.7, 1.4)  # Width/height ratio for faces
    
    def _init_license_plate_classifier(self):
        """Initialize license plate classification."""
        # License plate characteristics
        self.plate_aspect_ratios = {
            'us_standard': (2.0, 6.0),      # US standard plates
            'eu_standard': (1.8, 2.8),      # EU standard plates
            'square': (0.8, 1.2)            # Square plates
        }
        self.min_plate_area = 1500
        self.max_plate_area = 50000
        
        # Text detection for plates
        self.text_detector = cv2.createLineSegmentDetector()
    
    def _init_document_classifier(self):
        """Initialize document classification."""
        self.min_doc_area = 5000
        self.doc_aspect_ratios = {
            'a4_portrait': (0.7, 0.8),     # A4 portrait
            'a4_landscape': (1.2, 1.5),    # A4 landscape
            'letter': (0.75, 0.85),        # US Letter
            'business_card': (1.5, 2.0)    # Business card
        }
    
    def _init_screen_classifier(self):
        """Initialize screen/display classification."""
        self.screen_aspect_ratios = {
            '16_9': (1.7, 1.8),            # 16:9 displays
            '4_3': (1.3, 1.35),            # 4:3 displays
            'phone': (0.4, 0.6),           # Phone screens
            'ultrawide': (2.0, 3.5)        # Ultrawide monitors
        }
        self.min_screen_area = 3000
    
    def classify_detection(self, roi: np.ndarray, bbox: BBox) -> Optional[EnhancedDetection]:
        """Classify a detection ROI for the specific category."""
        if self.category == 'face':
            return self._classify_face(roi, bbox)
        elif self.category == 'license_plate':
            return self._classify_license_plate(roi, bbox)
        elif self.category == 'document':
            return self._classify_document(roi, bbox)
        elif self.category == 'screen':
            return self._classify_screen(roi, bbox)
        return None
    
    def _classify_face(self, roi: np.ndarray, bbox: BBox) -> Optional[EnhancedDetection]:
        """Enhanced face classification."""
        if roi.size == 0:
            return None
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Check basic face constraints - be more lenient
        if w < 20 or h < 20:  # Reduced minimum size
            return None
        
        aspect_ratio = w / h
        if not (0.5 <= aspect_ratio <= 2.0):  # More lenient aspect ratio
            return None
        
        # Multi-cascade detection for better accuracy
        frontal_faces = self.face_cascade_frontal.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)  # More sensitive
        )
        profile_faces = self.face_cascade_profile.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)  # More sensitive
        )
        
        confidence = 0.0
        sub_category = 'unknown'
        
        if len(frontal_faces) > 0:
            confidence = 0.85
            sub_category = 'frontal_face'
        elif len(profile_faces) > 0:
            confidence = 0.75
            sub_category = 'profile_face'
        else:
            # Additional heuristics for face-like regions
            confidence = self._analyze_face_features(gray)
            if confidence > 0.4:  # Lower threshold
                sub_category = 'partial_face'
        
        # Lower confidence threshold for faces
        if confidence < 0.5:  # Reduced from self.confidence_threshold
            return None
        
        # Calculate additional features
        features = {
            'aspect_ratio': aspect_ratio,
            'size_score': min(w * h / 10000, 1.0),  # Normalized size score
            'contrast': np.std(gray) / 255.0,       # Contrast measure
        }
        
        return EnhancedDetection(
            bbox=bbox,
            category='face',
            confidence=confidence,
            sub_category=sub_category,
            features=features,
            label=f"Face ({sub_category}) {confidence:.2f}"
        )
    
    def _classify_license_plate(self, roi: np.ndarray, bbox: BBox) -> Optional[EnhancedDetection]:
        """Enhanced license plate classification."""
        if roi.size == 0:
            return None
        
        h, w = roi.shape[:2]
        area = w * h
        aspect_ratio = w / h
        
        # Check size constraints - be more strict
        if area < 2000 or area > 30000:  # Stricter area limits
            return None
        
        # Check aspect ratio against known plate formats
        plate_type = None
        confidence = 0.0
        
        for plate_format, (min_ratio, max_ratio) in self.plate_aspect_ratios.items():
            if min_ratio <= aspect_ratio <= max_ratio:
                plate_type = plate_format
                confidence = 0.6  # Lower initial confidence
                break
        
        if plate_type is None:
            return None
        
        # Analyze text-like patterns - more strict
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text_confidence = self._analyze_text_patterns(gray)
        
        # Require minimum text confidence for license plates
        if text_confidence < 0.3:
            return None
        
        # Check for plate-like characteristics
        plate_characteristics = self._analyze_plate_characteristics(gray)
        
        # Combine confidences with stricter requirements
        final_confidence = (confidence + text_confidence + plate_characteristics) / 3
        
        if final_confidence < 0.65:  # Higher threshold for plates
            return None
        
        features = {
            'aspect_ratio': aspect_ratio,
            'area': area,
            'text_confidence': text_confidence,
            'plate_characteristics': plate_characteristics,
            'plate_type_score': confidence
        }
        
        return EnhancedDetection(
            bbox=bbox,
            category='license_plate',
            confidence=final_confidence,
            sub_category=plate_type,
            features=features,
            label=f"License Plate ({plate_type}) {final_confidence:.2f}"
        )
    
    def _classify_document(self, roi: np.ndarray, bbox: BBox) -> Optional[EnhancedDetection]:
        """Enhanced document classification."""
        if roi.size == 0:
            return None
        
        h, w = roi.shape[:2]
        area = w * h
        aspect_ratio = w / h
        
        if area < self.min_doc_area:
            return None
        
        # Check against known document formats
        doc_type = None
        confidence = 0.0
        
        for doc_format, (min_ratio, max_ratio) in self.doc_aspect_ratios.items():
            if min_ratio <= aspect_ratio <= max_ratio:
                doc_type = doc_format
                confidence = 0.6
                break
        
        if doc_type is None:
            doc_type = 'unknown_document'
            confidence = 0.5
        
        # Analyze document-like features
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Check for text lines (documents typically have horizontal text)
        text_score = self._analyze_document_text(gray)
        
        # Check for rectangular structure
        structure_score = self._analyze_document_structure(gray)
        
        # Combine scores
        final_confidence = (confidence + text_score + structure_score) / 3
        
        if final_confidence < 0.5:  # Lower threshold for documents
            return None
        
        features = {
            'aspect_ratio': aspect_ratio,
            'area': area,
            'text_score': text_score,
            'structure_score': structure_score
        }
        
        return EnhancedDetection(
            bbox=bbox,
            category='document',
            confidence=final_confidence,
            sub_category=doc_type,
            features=features,
            label=f"Document ({doc_type}) {final_confidence:.2f}"
        )
    
    def _classify_screen(self, roi: np.ndarray, bbox: BBox) -> Optional[EnhancedDetection]:
        """Enhanced screen/display classification."""
        if roi.size == 0:
            return None
        
        h, w = roi.shape[:2]
        area = w * h
        aspect_ratio = w / h
        
        if area < self.min_screen_area:
            return None
        
        # Check against known screen ratios
        screen_type = None
        confidence = 0.0
        
        for screen_format, (min_ratio, max_ratio) in self.screen_aspect_ratios.items():
            if min_ratio <= aspect_ratio <= max_ratio:
                screen_type = screen_format
                confidence = 0.7
                break
        
        if screen_type is None:
            return None
        
        # Analyze screen-like features
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Screens often have uniform brightness and sharp edges
        uniformity_score = self._analyze_screen_uniformity(gray)
        edge_score = self._analyze_screen_edges(gray)
        
        final_confidence = (confidence + uniformity_score + edge_score) / 3
        
        if final_confidence < self.confidence_threshold:
            return None
        
        features = {
            'aspect_ratio': aspect_ratio,
            'area': area,
            'uniformity_score': uniformity_score,
            'edge_score': edge_score
        }
        
        return EnhancedDetection(
            bbox=bbox,
            category='screen',
            confidence=final_confidence,
            sub_category=screen_type,
            features=features,
            label=f"Screen ({screen_type}) {final_confidence:.2f}"
        )
    
    def _analyze_face_features(self, gray: np.ndarray) -> float:
        """Analyze face-like features in grayscale image."""
        # Simple face feature analysis
        h, w = gray.shape
        
        # Check for eye-like regions (darker areas in upper half)
        upper_half = gray[:h//2, :]
        eye_regions = cv2.HoughCircles(
            upper_half, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=15
        )
        
        eye_score = 0.3 if eye_regions is not None and len(eye_regions[0]) >= 2 else 0.0
        
        # Check for mouth-like region (darker area in lower half)
        lower_half = gray[h//2:, :]
        mouth_score = 0.2 if np.mean(lower_half[h//4:3*h//4, w//4:3*w//4]) < np.mean(gray) else 0.0
        
        return eye_score + mouth_score
    
    def _analyze_text_patterns(self, gray: np.ndarray) -> float:
        """Analyze text-like patterns in image."""
        # Edge detection for text
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for horizontal lines (typical in license plates)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
        
        if lines is None:
            return 0.0
        
        # Count horizontal lines
        horizontal_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 15 or angle > 165:  # Nearly horizontal
                horizontal_lines += 1
        
        return min(horizontal_lines / 5.0, 1.0)  # Normalize to 0-1
    
    def _analyze_plate_characteristics(self, gray: np.ndarray) -> float:
        """Analyze license plate specific characteristics."""
        # Check for rectangular border
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Find the largest contour (should be the plate border)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if it's roughly rectangular
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        rect_score = 0.8 if len(approx) == 4 else 0.3
        
        # Check for uniform background (plates usually have consistent background)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        uniformity_score = max(0, 1.0 - std_intensity / 100.0)
        
        # Check for high contrast (text on background)
        contrast_score = min(std_intensity / 50.0, 1.0)
        
        return (rect_score + uniformity_score + contrast_score) / 3
    
    def _analyze_document_text(self, gray: np.ndarray) -> float:
        """Analyze document text patterns."""
        # Look for multiple horizontal text lines
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=20, maxLineGap=10)
        
        if lines is None:
            return 0.0
        
        horizontal_lines = sum(1 for line in lines 
                             if abs(np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0]) * 180 / np.pi) < 10)
        
        return min(horizontal_lines / 10.0, 1.0)
    
    def _analyze_document_structure(self, gray: np.ndarray) -> float:
        """Analyze document structural features."""
        # Check for rectangular structure
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if it's roughly rectangular
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Documents should have 4 corners (rectangular)
        if len(approx) == 4:
            return 0.8
        elif len(approx) in [3, 5, 6]:  # Close to rectangular
            return 0.5
        else:
            return 0.2
    
    def _analyze_screen_uniformity(self, gray: np.ndarray) -> float:
        """Analyze screen uniformity (screens often have consistent brightness)."""
        # Calculate local standard deviation
        kernel = np.ones((5, 5), np.float32) / 25
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_diff = (gray.astype(np.float32) - mean) ** 2
        local_std = np.sqrt(cv2.filter2D(sqr_diff, -1, kernel))
        
        # Screens should have relatively low local variation
        avg_std = np.mean(local_std)
        uniformity = max(0, 1.0 - avg_std / 50.0)  # Normalize
        
        return uniformity
    
    def _analyze_screen_edges(self, gray: np.ndarray) -> float:
        """Analyze screen edge characteristics."""
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for strong rectangular edges
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return 0.0
        
        # Count horizontal and vertical lines
        horizontal = sum(1 for line in lines 
                        if abs(np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0]) * 180 / np.pi) < 10)
        vertical = sum(1 for line in lines 
                      if abs(abs(np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0]) * 180 / np.pi) - 90) < 10)
        
        # Screens should have both horizontal and vertical edges
        edge_score = min((horizontal + vertical) / 8.0, 1.0)
        return edge_score


class EnhancedPrivacyDetector:
    """Enhanced privacy detector with improved accuracy and FPS."""
    
    def __init__(self):
        """Initialize enhanced detector."""
        self.specialized_classifiers = {
            'face': SpecializedClassifier('face'),
            'license_plate': SpecializedClassifier('license_plate'),
            'document': SpecializedClassifier('document'),
            'screen': SpecializedClassifier('screen')
        }
        
        # Performance optimizations
        self.frame_cache = {}
        self.cache_size_limit = 100
        self.skip_frame_threshold = 0.95  # Skip similar frames
        
        # Multi-threading for parallel classification (disabled for performance)
        self.thread_pool = None
        
        # FPS optimization settings
        self.adaptive_processing = True
        self.target_fps = 60
        self.processing_time_budget = 1.0 / self.target_fps  # 16.67ms for 60 FPS
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'cache_hits': 0,
            'avg_processing_time': 0.0,
            'classification_accuracy': 0.0,
            'fps_actual': 0.0
        }
        
        logger.info("Enhanced privacy detector initialized with specialized classifiers")
    
    def detect_privacy_violations(self, frame: np.ndarray) -> List[EnhancedDetection]:
        """Detect privacy violations with enhanced accuracy."""
        if frame is None or frame.size == 0:
            return []
        
        start_time = time.time()
        
        # Check frame cache for similar frames (FPS optimization)
        frame_hash = self._compute_frame_hash(frame)
        if self.adaptive_processing and frame_hash in self.frame_cache:
            self.stats['cache_hits'] += 1
            return self.frame_cache[frame_hash]
        
        # Initial detection using optimized OpenCV methods
        initial_detections = self._fast_initial_detection(frame)
        
        # Parallel classification of detected regions
        enhanced_detections = self._parallel_classification(frame, initial_detections)
        
        # Post-processing and filtering
        final_detections = self._post_process_detections(enhanced_detections)
        
        # Update cache
        if len(self.frame_cache) < self.cache_size_limit:
            self.frame_cache[frame_hash] = final_detections
        
        # Update statistics
        processing_time = time.time() - start_time
        self._update_stats(processing_time, len(final_detections))
        
        return final_detections
    
    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        """Compute hash for frame similarity detection."""
        # Downsample frame for hash computation
        small_frame = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute perceptual hash
        hash_bytes = hashlib.md5(gray.tobytes()).hexdigest()[:16]
        return hash_bytes
    
    def _fast_initial_detection(self, frame: np.ndarray) -> List[Dict]:
        """Fast initial detection to find candidate regions."""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Single scale for performance - multi-scale was too slow
        scale = 1.0
        
        # Face detection (fastest first)
        face_detections = self._detect_faces_fast(gray, scale)
        detections.extend(face_detections)
        
        # License plate detection
        plate_detections = self._detect_plates_fast(gray, scale)
        detections.extend(plate_detections)
        
        # Document detection
        doc_detections = self._detect_documents_fast(gray, scale)
        detections.extend(doc_detections)
        
        # Screen detection
        screen_detections = self._detect_screens_fast(gray, scale)
        detections.extend(screen_detections)
        
        return detections
    
    def _detect_faces_fast(self, gray: np.ndarray, scale: float = 1.0) -> List[Dict]:
        """Fast face detection."""
        detections = []
        
        # Use optimized cascade parameters for speed
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,  # Larger scale factor for speed
            minNeighbors=2,   # Fewer neighbors for speed
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_DO_CANNY_PRUNING
        )
        
        for (x, y, w, h) in faces:
            # Scale back to original coordinates
            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
            
            bbox = BBox(x1=x, y1=y, x2=x+w, y2=y+h, confidence=0.8)
            detections.append({
                'bbox': bbox,
                'category': 'face',
                'initial_confidence': 0.8
            })
        
        return detections
    
    def _detect_plates_fast(self, gray: np.ndarray, scale: float = 1.0) -> List[Dict]:
        """Fast license plate detection."""
        detections = []
        
        # Edge-based detection for rectangular regions
        edges = cv2.Canny(gray, 100, 200)  # Higher thresholds for cleaner edges
        
        # Morphological operations to connect text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))  # Horizontal kernel
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Skip small contours
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Stricter aspect ratio filter for license plates
            aspect_ratio = w / h
            if 2.0 <= aspect_ratio <= 5.0 and w > 80 and h > 20 and w < 400 and h < 100:
                # Scale back to original coordinates
                x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                
                bbox = BBox(x1=x, y1=y, x2=x+w, y2=y+h, confidence=0.6)
                detections.append({
                    'bbox': bbox,
                    'category': 'license_plate',
                    'initial_confidence': 0.6
                })
        
        return detections
    
    def _detect_documents_fast(self, gray: np.ndarray, scale: float = 1.0) -> List[Dict]:
        """Fast document detection."""
        detections = []
        
        # Adaptive threshold for document-like regions
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Find large rectangular regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 3000:  # Minimum document area
                x, y, w, h = cv2.boundingRect(contour)
                
                # Basic aspect ratio check
                aspect_ratio = w / h
                if 0.5 <= aspect_ratio <= 2.0:
                    # Scale back to original coordinates
                    x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                    
                    bbox = BBox(x1=x, y1=y, x2=x+w, y2=y+h, confidence=0.5)
                    detections.append({
                        'bbox': bbox,
                        'category': 'document',
                        'initial_confidence': 0.5
                    })
        
        return detections
    
    def _detect_screens_fast(self, gray: np.ndarray, scale: float = 1.0) -> List[Dict]:
        """Fast screen/display detection."""
        detections = []
        
        # Look for rectangular regions with uniform brightness
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Find rectangular contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Minimum screen area
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Look for rectangular shapes (4 corners)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio for common screen ratios
                    aspect_ratio = w / h
                    if 0.4 <= aspect_ratio <= 3.5:
                        # Scale back to original coordinates
                        x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                        
                        bbox = BBox(x1=x, y1=y, x2=x+w, y2=y+h, confidence=0.6)
                        detections.append({
                            'bbox': bbox,
                            'category': 'screen',
                            'initial_confidence': 0.6
                        })
        
        return detections
    
    def _parallel_classification(self, frame: np.ndarray, initial_detections: List[Dict]) -> List[EnhancedDetection]:
        """Classify detections with optimized performance."""
        if not initial_detections:
            return []
        
        enhanced_detections = []
        
        # Process sequentially for better performance (threading overhead was too high)
        for detection in initial_detections:
            bbox = detection['bbox']
            category = detection['category']
            
            # Extract ROI
            roi = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            
            # Classify directly
            result = self._classify_single_detection(roi, bbox, category)
            if result:
                enhanced_detections.append(result)
        
        return enhanced_detections
    
    def _classify_single_detection(self, roi: np.ndarray, bbox: BBox, category: str) -> Optional[EnhancedDetection]:
        """Classify a single detection."""
        if category in self.specialized_classifiers:
            classifier = self.specialized_classifiers[category]
            return classifier.classify_detection(roi, bbox)
        return None
    
    def _post_process_detections(self, detections: List[EnhancedDetection]) -> List[EnhancedDetection]:
        """Post-process detections to remove duplicates and improve accuracy."""
        if not detections:
            return []
        
        # Group by category for category-specific NMS
        category_groups = {}
        for detection in detections:
            if detection.category not in category_groups:
                category_groups[detection.category] = []
            category_groups[detection.category].append(detection)
        
        final_detections = []
        
        # Apply NMS per category
        for category, group_detections in category_groups.items():
            if len(group_detections) <= 1:
                final_detections.extend(group_detections)
                continue
            
            # Convert to format for NMS
            boxes = []
            confidences = []
            
            for detection in group_detections:
                boxes.append([
                    detection.bbox.x1,
                    detection.bbox.y1,
                    detection.bbox.width,
                    detection.bbox.height
                ])
                confidences.append(detection.confidence)
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    final_detections.append(group_detections[i])
        
        # Sort by confidence
        final_detections.sort(key=lambda x: x.confidence, reverse=True)
        
        return final_detections
    
    def _update_stats(self, processing_time: float, num_detections: int):
        """Update performance statistics."""
        self.stats['frames_processed'] += 1
        
        # Update moving average of processing time
        alpha = 0.1
        self.stats['avg_processing_time'] = (
            alpha * processing_time + (1 - alpha) * self.stats['avg_processing_time']
        )
        
        # Calculate actual FPS
        if processing_time > 0:
            current_fps = 1.0 / processing_time
            self.stats['fps_actual'] = (
                alpha * current_fps + (1 - alpha) * self.stats['fps_actual']
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_hit_rate = (
            self.stats['cache_hits'] / max(self.stats['frames_processed'], 1)
        )
        
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'target_fps': self.target_fps,
            'processing_time_budget': self.processing_time_budget,
            'performance_ratio': self.stats['fps_actual'] / max(self.target_fps, 1)
        }
    
    def set_target_fps(self, fps: int):
        """Set target FPS for performance optimization."""
        self.target_fps = fps
        self.processing_time_budget = 1.0 / fps
        logger.info(f"Target FPS set to {fps}, processing budget: {self.processing_time_budget*1000:.1f}ms")
    
    def set_adaptive_processing(self, enabled: bool):
        """Enable/disable adaptive processing optimizations."""
        self.adaptive_processing = enabled
        logger.info(f"Adaptive processing {'enabled' if enabled else 'disabled'}")
    
    def draw_enhanced_detections(self, frame: np.ndarray, detections: List[EnhancedDetection]) -> np.ndarray:
        """Draw enhanced detections with detailed information."""
        result = frame.copy()
        
        # Enhanced color scheme
        colors = {
            'face': (0, 255, 0),           # Green
            'license_plate': (0, 0, 255),  # Red
            'document': (255, 0, 0),       # Blue
            'screen': (255, 255, 0),       # Cyan
        }
        
        for detection in detections:
            color = colors.get(detection.category, (255, 255, 255))
            
            # Draw thicker box for higher confidence
            thickness = max(2, int(detection.confidence * 4))
            
            # Draw bounding box
            cv2.rectangle(
                result,
                (detection.bbox.x1, detection.bbox.y1),
                (detection.bbox.x2, detection.bbox.y2),
                color,
                thickness
            )
            
            # Enhanced label with sub-category and confidence
            if detection.label:
                label = detection.label
            else:
                sub_cat = f" ({detection.sub_category})" if detection.sub_category else ""
                label = f"{detection.category}{sub_cat} {detection.confidence:.2f}"
            
            # Multi-line label for detailed info
            label_lines = [label]
            if detection.features:
                # Add key features to label
                if 'aspect_ratio' in detection.features:
                    label_lines.append(f"AR: {detection.features['aspect_ratio']:.2f}")
            
            # Draw label background and text
            y_offset = detection.bbox.y1
            for i, line in enumerate(label_lines):
                label_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # Draw background
                cv2.rectangle(
                    result,
                    (detection.bbox.x1, y_offset - 20 - i * 15),
                    (detection.bbox.x1 + label_size[0] + 10, y_offset - i * 15),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    result,
                    line,
                    (detection.bbox.x1 + 5, y_offset - 5 - i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
        
        return result
    
    def cleanup(self):
        """Cleanup resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
        self.frame_cache.clear()
        logger.info("Enhanced privacy detector cleaned up")