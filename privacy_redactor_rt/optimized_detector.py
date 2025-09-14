"""Optimized privacy detection with primary/secondary face detection and improved performance."""

import logging
import time
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque

from .types import BBox

logger = logging.getLogger(__name__)


@dataclass
class OptimizedDetection:
    """Optimized privacy detection result."""
    bbox: BBox
    category: str
    confidence: float
    sub_category: Optional[str] = None
    priority: str = 'normal'  # 'primary', 'secondary', 'normal'
    features: Optional[Dict[str, float]] = None
    label: Optional[str] = None


class OptimizedPrivacyDetector:
    """Highly optimized privacy detector with primary/secondary face detection."""
    
    def __init__(self):
        """Initialize optimized detector."""
        # Load optimized face cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        
        # Performance settings
        self.target_fps = 60
        self.processing_budget = 1.0 / 60  # 16.67ms
        
        # Detection thresholds
        self.face_confidence_threshold = 0.6
        self.plate_confidence_threshold = 0.7
        self.document_confidence_threshold = 0.5
        self.screen_confidence_threshold = 0.6
        
        # Frame caching for performance
        self.frame_cache = {}
        self.cache_size_limit = 50
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'avg_processing_time': 0.0,
            'cache_hits': 0,
            'primary_faces': 0,
            'secondary_faces': 0,
            'total_detections': 0
        }
        
        logger.info("Optimized privacy detector initialized")
    
    def detect_privacy_violations(self, frame: np.ndarray) -> List[OptimizedDetection]:
        """Detect privacy violations with optimized performance."""
        if frame is None or frame.size == 0:
            return []
        
        start_time = time.time()
        
        # Check cache first
        frame_hash = self._compute_frame_hash(frame)
        if frame_hash in self.frame_cache:
            self.stats['cache_hits'] += 1
            return self.frame_cache[frame_hash]
        
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Primary and secondary face detection (highest priority)
        face_detections = self._detect_faces_optimized(gray, frame)
        detections.extend(face_detections)
        
        # 2. License plate detection (high priority)
        plate_detections = self._detect_plates_optimized(gray)
        detections.extend(plate_detections)
        
        # 3. Document detection (medium priority)
        doc_detections = self._detect_documents_optimized(gray)
        detections.extend(doc_detections)
        
        # 4. Screen detection (lower priority)
        screen_detections = self._detect_screens_optimized(gray)
        detections.extend(screen_detections)
        
        # Post-process and filter
        final_detections = self._post_process_optimized(detections)
        
        # Update cache
        if len(self.frame_cache) < self.cache_size_limit:
            self.frame_cache[frame_hash] = final_detections
        
        # Update statistics
        processing_time = time.time() - start_time
        self._update_stats(processing_time, final_detections)
        
        return final_detections
    
    def _detect_faces_optimized(self, gray: np.ndarray, frame: np.ndarray) -> List[OptimizedDetection]:
        """Optimized face detection with primary/secondary classification."""
        detections = []
        
        # Fast frontal face detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_DO_CANNY_PRUNING
        )
        
        # Profile face detection (if no frontal faces found or for completeness)
        profile_faces = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_DO_CANNY_PRUNING
        )
        
        all_faces = []
        
        # Process frontal faces
        for (x, y, w, h) in faces:
            bbox = BBox(x1=x, y1=y, x2=x+w, y2=y+h, confidence=0.85)
            all_faces.append({
                'bbox': bbox,
                'type': 'frontal',
                'confidence': 0.85,
                'area': w * h,
                'center': (x + w//2, y + h//2)
            })
        
        # Process profile faces
        for (x, y, w, h) in profile_faces:
            bbox = BBox(x1=x, y1=y, x2=x+w, y2=y+h, confidence=0.75)
            all_faces.append({
                'bbox': bbox,
                'type': 'profile',
                'confidence': 0.75,
                'area': w * h,
                'center': (x + w//2, y + h//2)
            })
        
        if not all_faces:
            return detections
        
        # Sort faces by area (largest first) to determine primary/secondary
        all_faces.sort(key=lambda f: f['area'], reverse=True)
        
        # Classify as primary/secondary
        for i, face in enumerate(all_faces):
            if i == 0:
                # Primary face (largest)
                priority = 'primary'
                sub_category = f'primary_{face["type"]}_face'
                self.stats['primary_faces'] += 1
            else:
                # Secondary faces
                priority = 'secondary'
                sub_category = f'secondary_{face["type"]}_face'
                self.stats['secondary_faces'] += 1
            
            # Additional face analysis
            roi = gray[face['bbox'].y1:face['bbox'].y2, face['bbox'].x1:face['bbox'].x2]
            face_quality = self._analyze_face_quality(roi)
            
            final_confidence = face['confidence'] * face_quality
            
            if final_confidence >= self.face_confidence_threshold:
                detection = OptimizedDetection(
                    bbox=face['bbox'],
                    category='face',
                    confidence=final_confidence,
                    sub_category=sub_category,
                    priority=priority,
                    features={
                        'area': face['area'],
                        'quality': face_quality,
                        'type': face['type']
                    },
                    label=f"Face ({priority} {face['type']}) {final_confidence:.2f}"
                )
                detections.append(detection)
        
        return detections
    
    def _analyze_face_quality(self, roi: np.ndarray) -> float:
        """Analyze face quality for better confidence scoring."""
        if roi.size == 0:
            return 0.0
        
        # Check contrast (good faces have good contrast)
        contrast = np.std(roi) / 255.0
        contrast_score = min(contrast * 2, 1.0)
        
        # Check sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500.0, 1.0)
        
        # Check size adequacy
        h, w = roi.shape
        size_score = min((w * h) / 2500.0, 1.0)  # Normalize to reasonable face size
        
        return (contrast_score + sharpness_score + size_score) / 3
    
    def _detect_plates_optimized(self, gray: np.ndarray) -> List[OptimizedDetection]:
        """Optimized license plate detection."""
        detections = []
        
        # Use adaptive threshold for better edge detection
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1500 or area > 25000:  # Size filter
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # License plate aspect ratio check
            if 2.0 <= aspect_ratio <= 5.5:
                # Additional validation
                roi = gray[y:y+h, x:x+w]
                plate_confidence = self._validate_license_plate(roi, aspect_ratio)
                
                if plate_confidence >= self.plate_confidence_threshold:
                    bbox = BBox(x1=x, y1=y, x2=x+w, y2=y+h, confidence=plate_confidence)
                    
                    # Determine plate type
                    if 2.0 <= aspect_ratio <= 2.8:
                        plate_type = 'eu_standard'
                    elif 2.8 <= aspect_ratio <= 4.5:
                        plate_type = 'us_standard'
                    else:
                        plate_type = 'custom'
                    
                    detection = OptimizedDetection(
                        bbox=bbox,
                        category='license_plate',
                        confidence=plate_confidence,
                        sub_category=plate_type,
                        features={
                            'aspect_ratio': aspect_ratio,
                            'area': area
                        },
                        label=f"License Plate ({plate_type}) {plate_confidence:.2f}"
                    )
                    detections.append(detection)
        
        return detections
    
    def _validate_license_plate(self, roi: np.ndarray, aspect_ratio: float) -> float:
        """Validate if ROI is actually a license plate."""
        if roi.size == 0:
            return 0.0
        
        # Check for text-like patterns
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=5, maxLineGap=3)
        
        text_score = 0.0
        if lines is not None:
            horizontal_lines = sum(1 for line in lines 
                                 if abs(np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0]) * 180 / np.pi) < 15)
            text_score = min(horizontal_lines / 3.0, 1.0)
        
        # Check uniformity (plates have relatively uniform background)
        uniformity = 1.0 - (np.std(roi) / 128.0)
        uniformity = max(0, uniformity)
        
        # Aspect ratio bonus
        aspect_bonus = 1.0 if 2.5 <= aspect_ratio <= 4.0 else 0.8
        
        return (text_score * 0.4 + uniformity * 0.4 + aspect_bonus * 0.2)
    
    def _detect_documents_optimized(self, gray: np.ndarray) -> List[OptimizedDetection]:
        """Optimized document detection."""
        detections = []
        
        # Use morphological operations to find document-like regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        edges = cv2.Canny(morph, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5000:  # Minimum document area
                continue
            
            # Approximate to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Documents should be roughly rectangular
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check for document-like aspect ratios
                doc_type = None
                confidence = 0.0
                
                if 0.7 <= aspect_ratio <= 0.8:
                    doc_type = 'a4_portrait'
                    confidence = 0.7
                elif 1.2 <= aspect_ratio <= 1.5:
                    doc_type = 'a4_landscape'
                    confidence = 0.7
                elif 1.5 <= aspect_ratio <= 2.0:
                    doc_type = 'business_card'
                    confidence = 0.6
                else:
                    doc_type = 'unknown_document'
                    confidence = 0.5
                
                if confidence >= self.document_confidence_threshold:
                    bbox = BBox(x1=x, y1=y, x2=x+w, y2=y+h, confidence=confidence)
                    
                    detection = OptimizedDetection(
                        bbox=bbox,
                        category='document',
                        confidence=confidence,
                        sub_category=doc_type,
                        features={
                            'aspect_ratio': aspect_ratio,
                            'area': area,
                            'corners': len(approx)
                        },
                        label=f"Document ({doc_type}) {confidence:.2f}"
                    )
                    detections.append(detection)
        
        return detections
    
    def _detect_screens_optimized(self, gray: np.ndarray) -> List[OptimizedDetection]:
        """Optimized screen/display detection."""
        detections = []
        
        # Use edge detection to find rectangular screens
        edges = cv2.Canny(gray, 100, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 3000:  # Minimum screen area
                continue
            
            # Approximate to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Screens should be rectangular
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check for common screen aspect ratios
                screen_type = None
                confidence = 0.0
                
                if 1.7 <= aspect_ratio <= 1.8:
                    screen_type = '16_9'
                    confidence = 0.8
                elif 1.3 <= aspect_ratio <= 1.35:
                    screen_type = '4_3'
                    confidence = 0.8
                elif 0.4 <= aspect_ratio <= 0.6:
                    screen_type = 'phone'
                    confidence = 0.7
                elif 2.0 <= aspect_ratio <= 3.5:
                    screen_type = 'ultrawide'
                    confidence = 0.7
                else:
                    screen_type = 'unknown_screen'
                    confidence = 0.6
                
                if confidence >= self.screen_confidence_threshold:
                    bbox = BBox(x1=x, y1=y, x2=x+w, y2=y+h, confidence=confidence)
                    
                    detection = OptimizedDetection(
                        bbox=bbox,
                        category='screen',
                        confidence=confidence,
                        sub_category=screen_type,
                        features={
                            'aspect_ratio': aspect_ratio,
                            'area': area
                        },
                        label=f"Screen ({screen_type}) {confidence:.2f}"
                    )
                    detections.append(detection)
        
        return detections
    
    def _post_process_optimized(self, detections: List[OptimizedDetection]) -> List[OptimizedDetection]:
        """Optimized post-processing with NMS."""
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
            
            # Apply NMS with category-specific thresholds
            nms_threshold = 0.3 if category == 'face' else 0.4
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, nms_threshold)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    final_detections.append(group_detections[i])
        
        # Sort by priority (primary faces first) then by confidence
        def sort_key(detection):
            priority_order = {'primary': 0, 'secondary': 1, 'normal': 2}
            return (priority_order.get(detection.priority, 2), -detection.confidence)
        
        final_detections.sort(key=sort_key)
        
        return final_detections
    
    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        """Compute simple frame hash for caching."""
        # Downsample for hash
        small = cv2.resize(frame, (32, 32))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return str(hash(gray.tobytes()))
    
    def _update_stats(self, processing_time: float, detections: List[OptimizedDetection]):
        """Update performance statistics."""
        self.stats['frames_processed'] += 1
        self.stats['total_detections'] += len(detections)
        
        # Update moving average
        alpha = 0.1
        self.stats['avg_processing_time'] = (
            alpha * processing_time + (1 - alpha) * self.stats['avg_processing_time']
        )
    
    def draw_detections(self, frame: np.ndarray, detections: List[OptimizedDetection]) -> np.ndarray:
        """Draw optimized detection results."""
        result = frame.copy()
        
        # Color scheme
        colors = {
            'face': (0, 255, 0),      # Green for faces
            'license_plate': (0, 0, 255),  # Red for plates
            'document': (255, 0, 0),   # Blue for documents
            'screen': (255, 255, 0)    # Cyan for screens
        }
        
        # Priority colors for faces
        priority_colors = {
            'primary': (0, 255, 0),    # Bright green
            'secondary': (0, 200, 0),  # Darker green
            'normal': (0, 150, 0)      # Even darker green
        }
        
        for detection in detections:
            # Choose color based on category and priority
            if detection.category == 'face' and detection.priority in priority_colors:
                color = priority_colors[detection.priority]
            else:
                color = colors.get(detection.category, (128, 128, 128))
            
            # Draw bounding box
            thickness = 3 if detection.priority == 'primary' else 2
            cv2.rectangle(
                result,
                (detection.bbox.x1, detection.bbox.y1),
                (detection.bbox.x2, detection.bbox.y2),
                color,
                thickness
            )
            
            # Draw label
            if detection.label:
                label_size = cv2.getTextSize(detection.label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background
                cv2.rectangle(
                    result,
                    (detection.bbox.x1, detection.bbox.y1 - 25),
                    (detection.bbox.x1 + label_size[0] + 10, detection.bbox.y1),
                    color,
                    -1
                )
                
                # Text
                cv2.putText(
                    result,
                    detection.label,
                    (detection.bbox.x1 + 5, detection.bbox.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        fps = 1.0 / max(self.stats['avg_processing_time'], 0.001)
        
        return {
            **self.stats,
            'fps': fps,
            'target_fps': self.target_fps,
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['frames_processed'], 1),
            'avg_detections_per_frame': self.stats['total_detections'] / max(self.stats['frames_processed'], 1)
        }
    
    def set_target_fps(self, fps: int):
        """Set target FPS."""
        self.target_fps = fps
        self.processing_budget = 1.0 / fps
        logger.info(f"Target FPS set to {fps}")
    
    def cleanup(self):
        """Clean up resources."""
        self.frame_cache.clear()
        logger.info("Optimized privacy detector cleaned up")