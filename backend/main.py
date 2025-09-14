import cv2
import numpy as np
import asyncio
import threading
from queue import Queue
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Choose your text detection method
TEXT_DETECTION_METHOD = "opencv"  # "opencv" or "paddle" or "simple"

if TEXT_DETECTION_METHOD == "paddle":
    try:
        from paddleocr import PaddleOCR
        HAS_PADDLE = True
    except ImportError:
        print("PaddleOCR not installed. Using OpenCV method.")
        HAS_PADDLE = False
        TEXT_DETECTION_METHOD = "opencv"
else:
    HAS_PADDLE = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        """Simple OpenCV-based text detection using contours"""
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
                # Filter by size (likely text regions)
                if w > 20 and h > 10 and h < 50:  # Adjust thresholds as needed
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

class OptimizedTextBlurTrack(VideoStreamTrack):
    def __init__(self, track, detection_interval=4, blur_strength=51):
        super().__init__()
        self.track = track
        self.detection_interval = detection_interval
        self.frame_count = 0
        self.cached_boxes = []
        self.blur_strength = blur_strength
        
        # Initialize text detector
        self.detector = TextDetector(method=TEXT_DETECTION_METHOD)
        
        # Async processing setup
        self.detection_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.processing_active = True
        
        # Start background detection thread
        self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self.detection_thread.start()
        
    def _detection_worker(self):
        """Background thread for text detection"""
        while self.processing_active:
            try:
                frame_data = self.detection_queue.get(timeout=0.1)
                if frame_data is None:  # Shutdown signal
                    break
                    
                boxes = self.detector.detect_text_regions(frame_data)
                
                # Only update if we have a reasonable number of detections
                if len(boxes) < 50:  # Prevent too many false positives
                    if not self.result_queue.full():
                        try:
                            self.result_queue.put(boxes, block=False)
                        except:
                            pass
                            
            except Exception as e:
                continue
    
    def apply_selective_blur(self, image, text_boxes):
        """Apply blur only to detected text regions"""
        if not text_boxes:
            return image
        
        try:
            # Create mask for text regions
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            for box in text_boxes:
                x, y, w, h = box
                # Add padding around detected text
                padding = 8
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2*padding)
                h = min(image.shape[0] - y, h + 2*padding)
                
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            
            # Apply Gaussian blur to entire image
            blurred = cv2.GaussianBlur(image, (self.blur_strength, self.blur_strength), 0)
            
            # Use mask to selectively apply blur
            result = np.where(mask[..., None] == 255, blurred, image)
            
            return result.astype(np.uint8)
            
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
                    self.cached_boxes = self.result_queue.get_nowait()
            except:
                pass
            
            # Apply selective blur
            processed_img = self.apply_selective_blur(img, self.cached_boxes)
            
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
                blur_track = OptimizedTextBlurTrack(
                    track, 
                    detection_interval=2,  # Process every 2nd frame (was 3)
                    blur_strength=41      # Stronger blur (was 31)
                )
                active_tracks.append(blur_track)
                pc.addTrack(blur_track)

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