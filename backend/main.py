import cv2
import numpy as np
import asyncio
import threading
from queue import Queue
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# Choose your detection methods
TEXT_DETECTION_METHOD = "opencv"  # "opencv" or "paddle" or "simple"
ENABLE_OBJECT_DETECTION = True    # Enable/disable YOLO object detection
DEBUG_MODE = False               # If True, show detection boxes; if False, blur secondary objects

if TEXT_DETECTION_METHOD == "paddle":
    try:
        from paddleocr import PaddleOCR
        HAS_PADDLE = True
    except ImportError:
        print("PaddleOCR not installed. Using OpenCV method.")
        HAS_PADDLE = False
        TEXT_DETECTION_METHOD = "opencv"
else:
    HAS_PADDLE = True

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

class ObjectDetector:
    def __init__(self):
        # Initialize YOLO model with YOLOv8n (smaller, faster model)
        self.model = YOLO('yolov8n.pt')
        
    def detect_objects(self, image):
        """Detect objects in the image using YOLO"""
        try:
            # Run inference
            results = self.model(image, conf=0.5)  # Confidence threshold of 0.5
            detections = []
            
            # Process results
            if results and len(results) > 0:
                result = results[0]  # Get first result
                for box in result.boxes:
                    # Get box coordinates, confidence and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    # Convert to x, y, w, h format
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)
                    
                    detections.append({
                        'box': (x, y, w, h),
                        'confidence': confidence,
                        'class': class_name
                    })
            
            return detections
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []

class TextDetector:
    def __init__(self, method="opencv"):
        self.method = method
        if method == "paddle" and HAS_PADDLE:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        elif method == "opencv":
            # You'll need to download frozen_east_text_detection.pb
            # Or use simple contour-based detection
            pass
    
    def detect_text_regions(self, image):
        """Detect text regions and return bounding boxes"""
        if self.method == "paddle" and HAS_PADDLE:
            return self._paddle_detect(image)
        elif self.method == "opencv":
            return self._opencv_detect(image)
        else:
            return self._simple_detect(image)
    
    def _paddle_detect(self, image):
        """PaddleOCR detection"""
        try:
            results = self.ocr.ocr(image, cls=True)
            boxes = []
            if results and results[0]:
                for line in results[0]:
                    if line:
                        box = np.array(line[0]).astype(int)
                        # Convert to x, y, w, h format
                        x_coords = box[:, 0]
                        y_coords = box[:, 1]
                        x, y = int(min(x_coords)), int(min(y_coords))
                        w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                        boxes.append((x, y, w, h))
            return boxes
        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return []
    
    def _opencv_detect(self, image):
        """Improved OpenCV-based text detection using contours and aspect ratio filtering"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply morphological operations to detect text-like regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

            # Threshold
            _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Connect horizontally oriented regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                # Stricter: wider, larger, and not too tall
                if w > 40 and h > 10 and h < 50 and aspect_ratio > 3.0 and area > 400:
                    boxes.append((x, y, w, h))

            return boxes
        except Exception as e:
            print(f"OpenCV detection error: {e}")
            return []
    
    def _simple_detect(self, image):
        """Very simple edge-based detection for testing"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Basic filtering
                if w > 15 and h > 8 and w/h > 2:  # Likely text aspect ratio
                    boxes.append((x, y, w, h))
            
            return boxes
        except Exception as e:
            print(f"Simple detection error: {e}")
            return []

class VideoProcessingTrack(VideoStreamTrack):
    def __init__(self, track, detection_interval=4, blur_strength=51):
        super().__init__()
        self.track = track
        self.detection_interval = detection_interval
        self.frame_count = 0
        self.cached_text_boxes = []
        self.cached_object_detections = []
        self.blur_strength = blur_strength
        
        # Initialize detectors
        self.text_detector = TextDetector(method=TEXT_DETECTION_METHOD)
        if ENABLE_OBJECT_DETECTION:
            self.object_detector = ObjectDetector()
        
        # Async processing setup
        self.detection_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.processing_active = True
        
        # Start background detection thread
        self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self.detection_thread.start()
        
    def _detection_worker(self):
        """Background thread for text and object detection"""
        while self.processing_active:
            try:
                frame_data = self.detection_queue.get(timeout=0.1)
                if frame_data is None:  # Shutdown signal
                    break
                
                results = {
                    'text_boxes': [],
                    'object_detections': []
                }
                
                # Text detection
                text_boxes = self.text_detector.detect_text_regions(frame_data)
                if len(text_boxes) < 50:  # Prevent too many false positives
                    results['text_boxes'] = text_boxes
                
                # Object detection
                if ENABLE_OBJECT_DETECTION:
                    object_detections = self.object_detector.detect_objects(frame_data)
                    results['object_detections'] = object_detections
                
                # Update results
                if not self.result_queue.full():
                    try:
                        self.result_queue.put(results, block=False)
                    except:
                        pass
                            
            except Exception as e:
                print(f"Detection worker error: {e}")
                continue
    
    def process_frame(self, image):
        """Process frame with text blur and object detection"""
        try:
            # Create a copy for drawing
            result_image = image.copy()
            
            # Apply text blur
            if self.cached_text_boxes:
                # Create mask for text regions
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                
                for box in self.cached_text_boxes:
                    x, y, w, h = box
                    # Add padding around detected text
                    padding = 8
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(image.shape[1] - x, w + 2*padding)
                    h = min(image.shape[0] - y, h + 2*padding)
                    
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                
                # Apply Gaussian blur to text regions
                blurred = cv2.GaussianBlur(image, (self.blur_strength, self.blur_strength), 0)
                result_image = np.where(mask[..., None] == 255, blurred, result_image)
            
                # Handle object detection
            if ENABLE_OBJECT_DETECTION and self.cached_object_detections:
                # Find primary person (largest person in frame)
                people = [d for d in self.cached_object_detections if d['class'] == 'person']
                people.sort(key=lambda x: x['box'][2] * x['box'][3], reverse=True)
                primary_person = people[0] if people else None

                if DEBUG_MODE:
                    # In debug mode, just draw detection boxes for all objects
                    for detection in self.cached_object_detections:
                        x, y, w, h = detection['box']
                        class_name = detection['class']
                        confidence = detection['confidence']
                        
                        # Color scheme
                        if class_name == 'person':
                            if detection == primary_person:  # Primary person
                                color = (0, 255, 0)     # Bright green
                            else:
                                color = (0, 100, 0)     # Dark green
                        elif class_name in ['book', 'paper', 'document', 'card']:
                            color = (255, 165, 0)       # Blue
                        else:
                            # Check for potential ID cards
                            x, y, w, h = detection['box']
                            aspect_ratio = w / h if h != 0 else 0
                            min_size = 50
                            is_id_shaped = (1.4 <= aspect_ratio <= 1.9) and w > min_size and h > min_size
                            
                            if is_id_shaped:
                                color = (255, 0, 0)     # Red for potential IDs
                            else:
                                color = (128, 128, 128)  # Gray for other objects
                        
                        # Draw bounding box
                        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                        
                        # Draw label
                        label = f"{class_name} {confidence:.2f}"
                        cv2.putText(result_image, label, (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    # In non-debug mode, blur everything except primary person and misc objects
                    blur_mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Start with no blur
                    
                    # Mark areas to blur (all people except primary, and all documents)
                    for detection in self.cached_object_detections:
                        x, y, w, h = detection['box']
                        class_name = detection['class']
                        
                        # Determine if this object should be blurred
                        should_blur = False
                        if class_name == 'person':
                            if detection != primary_person:  # Blur secondary people
                                should_blur = True
                        elif class_name in ['book', 'paper', 'document', 'card']:  # Blur documents
                            should_blur = True
                        else:
                            # Check for potential ID cards based on aspect ratio and size
                            x, y, w, h = detection['box']
                            aspect_ratio = w / h if h != 0 else 0
                            min_size = 50  # Minimum size in pixels
                            
                            # Typical ID card aspect ratio is around 1.6 (like credit cards)
                            # Allow some variation in the ratio
                            is_id_shaped = (1.4 <= aspect_ratio <= 1.9) and w > min_size and h > min_size
                            
                            if is_id_shaped:
                                should_blur = True
                        # Misc objects are not marked for blur
                        
                        if should_blur:
                            # Add padding for better privacy
                            padding = 10
                            x1 = max(0, x - padding)
                            y1 = max(0, y - padding)
                            x2 = min(image.shape[1], x + w + padding)
                            y2 = min(image.shape[0], y + h + padding)
                            blur_mask[y1:y2, x1:x2] = 255
                    
                    # Apply blur where mask is set
                    if np.any(blur_mask):  # Only if there's something to blur
                        blurred = cv2.GaussianBlur(result_image, (self.blur_strength, self.blur_strength), 0)
                        result_image = np.where(blur_mask[..., None] == 255, blurred, result_image)
            
            return result_image.astype(np.uint8)
            
        except Exception as e:
            print(f"Blur application error: {e}")
            return image
    
    async def recv(self):
        try:
            frame = await self.track.recv()
            img = frame.to_ndarray(format="bgr24")
            
            # Send frame for background detection (non-blocking)
            if self.frame_count % self.detection_interval == 0:
                if not self.detection_queue.full():
                    try:
                        self.detection_queue.put(img.copy(), block=False)
                    except:
                        pass
            
            # Get latest detection results (non-blocking)
            try:
                while not self.result_queue.empty():
                    results = self.result_queue.get_nowait()
                    self.cached_text_boxes = results['text_boxes']
                    self.cached_object_detections = results['object_detections']
            except:
                pass
            
            # Process frame with detections
            processed_img = self.process_frame(img)
            
            self.frame_count += 1
            
            # Create new frame
            new_frame = VideoFrame.from_ndarray(processed_img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            
            return new_frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            # Return original frame on error
            frame = await self.track.recv()
            return frame
    
    def cleanup(self):
        """Clean up resources"""
        self.processing_active = False
        try:
            self.detection_queue.put(None, timeout=1)  # Shutdown signal
        except:
            pass

# Store active tracks for cleanup
active_tracks = []

@app.post("/offer")
async def offer(request: Request):
    """Accepts a WebRTC offer and returns an answer with text-blurred video."""
    try:
        data = await request.json()
        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

        pc = RTCPeerConnection()

        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                processed_track = VideoProcessingTrack(
                    track, 
                    detection_interval=2,  # Process every 2nd frame
                    blur_strength=41      # Blur strength for text regions
                )
                active_tracks.append(processed_track)
                pc.addTrack(processed_track)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState in ["failed", "closed"]:
                # Cleanup tracks
                for track in active_tracks:
                    if hasattr(track, 'cleanup'):
                        track.cleanup()
                active_tracks.clear()

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
        
    except Exception as e:
        print(f"WebRTC error: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    print(f"Starting server with {TEXT_DETECTION_METHOD} text detection...")
    uvicorn.run(app, host="0.0.0.0", port=8000)