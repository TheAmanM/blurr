"""ONNX-based privacy violation detection with CoreML acceleration."""

import logging
import time
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False

from .types import BBox

logger = logging.getLogger(__name__)


@dataclass
class PrivacyDetection:
    """Privacy violation detection result."""
    bbox: BBox
    category: str  # 'face', 'license_plate', 'document', 'screen', 'id_card', 'credit_card'
    confidence: float
    label: Optional[str] = None


class ONNXPrivacyDetector:
    """High-performance privacy violation detector using ONNX Runtime."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize ONNX privacy detector.
        
        Args:
            model_path: Path to ONNX model file. If None, uses default YOLOv8 model.
        """
        self.model_path = model_path
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_names: Optional[List[str]] = None
        self.input_shape: Optional[Tuple[int, ...]] = None
        
        # Model configuration
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.input_size = (640, 640)  # Standard YOLO input size
        
        # Privacy categories mapping
        self.privacy_categories = {
            0: 'face',
            1: 'license_plate', 
            2: 'document',
            3: 'screen',
            4: 'id_card',
            5: 'credit_card',
            6: 'phone_number',
            7: 'email',
            8: 'address'
        }
        
        # Performance tracking
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'avg_inference_time': 0.0,
            'avg_preprocessing_time': 0.0,
            'avg_postprocessing_time': 0.0
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize ONNX Runtime session with optimal providers."""
        if not ONNX_AVAILABLE:
            logger.warning("ONNX Runtime not available. Using OpenCV fallback detection.")
            self.model_path = "FALLBACK_OPENCV"
            return
        
        # Determine best execution provider
        providers = self._get_optimal_providers()
        logger.info(f"Initializing ONNX Runtime with providers: {providers}")
        
        try:
            # Use default YOLOv8 model if no custom model provided
            if self.model_path is None:
                self.model_path = self._download_default_model()
            
            # Create inference session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            # Get model input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.input_shape = self.session.get_inputs()[0].shape
            
            logger.info(f"ONNX model loaded successfully")
            logger.info(f"Input shape: {self.input_shape}")
            logger.info(f"Output names: {self.output_names}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX model: {e}")
            raise RuntimeError(f"ONNX model initialization failed: {e}")
    
    def _get_optimal_providers(self) -> List[str]:
        """Get optimal execution providers based on available hardware."""
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX providers: {available_providers}")
        
        # Priority order: CoreML > CUDA > CPU
        preferred_providers = []
        
        # CoreML for Apple Silicon/Intel Macs
        if 'CoreMLExecutionProvider' in available_providers:
            preferred_providers.append('CoreMLExecutionProvider')
            logger.info("Using CoreML acceleration")
        
        # CUDA for NVIDIA GPUs
        elif 'CUDAExecutionProvider' in available_providers:
            preferred_providers.append('CUDAExecutionProvider')
            logger.info("Using CUDA acceleration")
        
        # Always include CPU as fallback
        preferred_providers.append('CPUExecutionProvider')
        
        return preferred_providers
    
    def _download_default_model(self) -> str:
        """Download or create a default privacy detection model."""
        # For now, we'll create a mock model path
        # In production, you'd download a pre-trained YOLOv8 model
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "privacy_yolov8.onnx"
        
        if not model_path.exists():
            # Create a simple mock model for demonstration
            # In production, download from model repository
            logger.warning("No privacy detection model found. Using CPU-based fallback.")
            return self._create_fallback_model(model_path)
        
        return str(model_path)
    
    def _create_fallback_model(self, model_path: Path) -> str:
        """Create a fallback detection system when no ONNX model is available."""
        logger.info("Creating fallback detection system")
        # Return a flag to use OpenCV-based detection
        return "FALLBACK_OPENCV"
    
    def detect_privacy_violations(self, frame: np.ndarray) -> List[PrivacyDetection]:
        """Detect privacy violations in a frame.
        
        Args:
            frame: Input frame as BGR numpy array
            
        Returns:
            List of privacy violation detections
        """
        if frame is None or frame.size == 0:
            return []
        
        start_time = time.time()
        
        try:
            # Use fallback detection if no ONNX model
            if self.model_path == "FALLBACK_OPENCV":
                return self._fallback_detect(frame)
            
            # Preprocess frame
            preprocess_start = time.time()
            input_tensor = self._preprocess_frame(frame)
            preprocess_time = time.time() - preprocess_start
            
            # Run inference
            inference_start = time.time()
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            inference_time = time.time() - inference_start
            
            # Post-process results
            postprocess_start = time.time()
            detections = self._postprocess_outputs(outputs, frame.shape)
            postprocess_time = time.time() - postprocess_start
            
            # Update statistics
            self._update_stats(inference_time, preprocess_time, postprocess_time, len(detections))
            
            return detections
            
        except Exception as e:
            logger.error(f"Privacy detection failed: {e}")
            return []
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for ONNX model input."""
        # Resize to model input size
        resized = cv2.resize(frame, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def _postprocess_outputs(self, outputs: List[np.ndarray], original_shape: Tuple[int, int, int]) -> List[PrivacyDetection]:
        """Post-process ONNX model outputs to extract detections."""
        detections = []
        
        if not outputs or len(outputs) == 0:
            return detections
        
        # Assuming YOLOv8 output format: [batch, num_detections, 85]
        # Where 85 = 4 (bbox) + 1 (confidence) + 80 (classes)
        predictions = outputs[0][0]  # Remove batch dimension
        
        # Scale factors for converting back to original image size
        scale_x = original_shape[1] / self.input_size[0]
        scale_y = original_shape[0] / self.input_size[1]
        
        for prediction in predictions:
            # Extract bbox and confidence
            x_center, y_center, width, height = prediction[:4]
            confidence = prediction[4]
            class_scores = prediction[5:]
            
            # Filter by confidence threshold
            if confidence < self.confidence_threshold:
                continue
            
            # Get class with highest score
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            
            # Only keep privacy-related classes
            if class_id not in self.privacy_categories:
                continue
            
            # Convert to absolute coordinates
            x1 = int((x_center - width / 2) * scale_x)
            y1 = int((y_center - height / 2) * scale_y)
            x2 = int((x_center + width / 2) * scale_x)
            y2 = int((y_center + height / 2) * scale_y)
            
            # Clamp to image bounds
            x1 = max(0, min(x1, original_shape[1] - 1))
            y1 = max(0, min(y1, original_shape[0] - 1))
            x2 = max(x1 + 1, min(x2, original_shape[1]))
            y2 = max(y1 + 1, min(y2, original_shape[0]))
            
            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(confidence))
            
            detection = PrivacyDetection(
                bbox=bbox,
                category=self.privacy_categories[class_id],
                confidence=float(class_confidence),
                label=f"{self.privacy_categories[class_id]} ({confidence:.2f})"
            )
            
            detections.append(detection)
        
        # Apply Non-Maximum Suppression
        detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections: List[PrivacyDetection]) -> List[PrivacyDetection]:
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if len(detections) <= 1:
            return detections
        
        # Convert to format expected by cv2.dnn.NMSBoxes
        boxes = []
        confidences = []
        
        for detection in detections:
            boxes.append([
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.width,
                detection.bbox.height
            ])
            confidences.append(detection.confidence)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            self.nms_threshold
        )
        
        # Return filtered detections
        if len(indices) > 0:
            return [detections[i] for i in indices.flatten()]
        else:
            return []
    
    def _fallback_detect(self, frame: np.ndarray) -> List[PrivacyDetection]:
        """Fallback detection using OpenCV when ONNX model is not available."""
        detections = []
        
        # Face detection using Haar cascades
        face_detections = self._detect_faces_opencv(frame)
        detections.extend(face_detections)
        
        # License plate detection (simple approach)
        plate_detections = self._detect_license_plates_opencv(frame)
        detections.extend(plate_detections)
        
        # Document detection (rectangular regions with text)
        doc_detections = self._detect_documents_opencv(frame)
        detections.extend(doc_detections)
        
        return detections
    
    def _detect_faces_opencv(self, frame: np.ndarray) -> List[PrivacyDetection]:
        """Detect faces using OpenCV Haar cascades."""
        detections = []
        
        try:
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                bbox = BBox(x1=x, y1=y, x2=x+w, y2=y+h, confidence=0.8)
                detection = PrivacyDetection(
                    bbox=bbox,
                    category='face',
                    confidence=0.8,
                    label='Face (OpenCV)'
                )
                detections.append(detection)
                
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
        
        return detections
    
    def _detect_license_plates_opencv(self, frame: np.ndarray) -> List[PrivacyDetection]:
        """Detect license plates using simple computer vision."""
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (license plates are typically wider than tall)
                aspect_ratio = w / h
                if 2.0 <= aspect_ratio <= 6.0 and w > 80 and h > 20:
                    bbox = BBox(x1=x, y1=y, x2=x+w, y2=y+h, confidence=0.6)
                    detection = PrivacyDetection(
                        bbox=bbox,
                        category='license_plate',
                        confidence=0.6,
                        label='License Plate (OpenCV)'
                    )
                    detections.append(detection)
                    
        except Exception as e:
            logger.warning(f"License plate detection failed: {e}")
        
        return detections
    
    def _detect_documents_opencv(self, frame: np.ndarray) -> List[PrivacyDetection]:
        """Detect documents using edge detection and contour analysis."""
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size (documents are typically large rectangular regions)
                if w > 100 and h > 100 and w * h > 10000:
                    # Check if it's roughly rectangular
                    area = cv2.contourArea(contour)
                    rect_area = w * h
                    
                    if area > 0 and (area / rect_area) > 0.7:
                        bbox = BBox(x1=x, y1=y, x2=x+w, y2=y+h, confidence=0.5)
                        detection = PrivacyDetection(
                            bbox=bbox,
                            category='document',
                            confidence=0.5,
                            label='Document (OpenCV)'
                        )
                        detections.append(detection)
                        
        except Exception as e:
            logger.warning(f"Document detection failed: {e}")
        
        return detections
    
    def _update_stats(self, inference_time: float, preprocess_time: float, 
                     postprocess_time: float, num_detections: int):
        """Update performance statistics."""
        self.stats['total_frames'] += 1
        self.stats['total_detections'] += num_detections
        
        # Update moving averages
        alpha = 0.1
        self.stats['avg_inference_time'] = (
            alpha * inference_time + (1 - alpha) * self.stats['avg_inference_time']
        )
        self.stats['avg_preprocessing_time'] = (
            alpha * preprocess_time + (1 - alpha) * self.stats['avg_preprocessing_time']
        )
        self.stats['avg_postprocessing_time'] = (
            alpha * postprocess_time + (1 - alpha) * self.stats['avg_postprocessing_time']
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_time = (
            self.stats['avg_inference_time'] + 
            self.stats['avg_preprocessing_time'] + 
            self.stats['avg_postprocessing_time']
        )
        
        fps = 1.0 / total_time if total_time > 0 else 0
        
        return {
            **self.stats,
            'avg_total_time': total_time,
            'estimated_fps': fps,
            'provider': self.session.get_providers()[0] if self.session else 'None'
        }
    
    def draw_detections(self, frame: np.ndarray, detections: List[PrivacyDetection]) -> np.ndarray:
        """Draw bounding boxes for privacy violations.
        
        Args:
            frame: Input frame
            detections: List of privacy detections
            
        Returns:
            Frame with bounding boxes drawn
        """
        result = frame.copy()
        
        # Color mapping for different privacy categories
        colors = {
            'face': (0, 255, 0),           # Green
            'license_plate': (0, 0, 255),  # Red
            'document': (255, 0, 0),       # Blue
            'screen': (255, 255, 0),       # Cyan
            'id_card': (255, 0, 255),      # Magenta
            'credit_card': (0, 255, 255),  # Yellow
            'phone_number': (128, 0, 128), # Purple
            'email': (255, 165, 0),        # Orange
            'address': (0, 128, 128)       # Teal
        }
        
        for detection in detections:
            color = colors.get(detection.category, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(
                result,
                (detection.bbox.x1, detection.bbox.y1),
                (detection.bbox.x2, detection.bbox.y2),
                color,
                2
            )
            
            # Draw label with background
            if detection.label:
                label_size = cv2.getTextSize(detection.label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(
                    result,
                    (detection.bbox.x1, detection.bbox.y1 - label_size[1] - 10),
                    (detection.bbox.x1 + label_size[0] + 10, detection.bbox.y1),
                    color,
                    -1
                )
                
                # Draw label text
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
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for detections."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold set to {self.confidence_threshold}")
    
    def set_nms_threshold(self, threshold: float):
        """Set NMS threshold for duplicate removal."""
        self.nms_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"NMS threshold set to {self.nms_threshold}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.session:
            del self.session
            self.session = None
        logger.info("ONNX detector cleaned up")