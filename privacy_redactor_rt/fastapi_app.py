"""FastAPI web interface with WebRTC for Privacy Redactor RT."""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Optional, Set

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from privacy_redactor_rt.config import Config, load_config
from privacy_redactor_rt.pipeline import RealtimePipeline

logger = logging.getLogger(__name__)


class ConfigUpdate(BaseModel):
    """Model for configuration updates."""
    categories: Optional[list] = None
    redaction_method: Optional[str] = None
    detector_stride: Optional[int] = None
    ocr_refresh_stride: Optional[int] = None


class FastAPIApp:
    """FastAPI-based web interface for privacy redaction."""
    
    def __init__(self):
        """Initialize the FastAPI application."""
        self.app = FastAPI(title="Privacy Redactor RT", version="1.0.0")
        self.config: Optional[Config] = None
        self.pipeline: Optional[RealtimePipeline] = None
        self.active_connections: Set[WebSocket] = set()
        self.connection_pipelines: Dict[str, RealtimePipeline] = {}
        
        self._load_config()
        self._setup_routes()
        self._setup_static_files()
    
    def _load_config(self):
        """Load configuration."""
        try:
            config_path = Path("default.yaml")
            if config_path.exists():
                self.config = load_config(config_path)
            else:
                self.config = Config()
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = Config()
    
    def _setup_static_files(self):
        """Setup static file serving."""
        # Create static directory if it doesn't exist
        static_dir = Path("static")
        static_dir.mkdir(exist_ok=True)
        
        # Create a simple HTML interface
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Privacy Redactor RT</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { display: flex; gap: 20px; }
        .video-section { flex: 2; }
        .controls-section { flex: 1; }
        video { width: 100%; max-width: 640px; }
        canvas { width: 100%; max-width: 640px; border: 1px solid #ccc; }
        .control-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        select, input { width: 100%; padding: 5px; }
        .stats { background: #f5f5f5; padding: 10px; border-radius: 5px; }
        .metric { display: flex; justify-content: space-between; margin-bottom: 5px; }
    </style>
</head>
<body>
    <h1>🔒 Privacy Redactor RT</h1>
    <p>Real-time sensitive information detection and redaction system</p>
    
    <div class="container">
        <div class="video-section">
            <h2>📺 Live Video Stream</h2>
            <video id="localVideo" autoplay muted></video>
            <canvas id="processedCanvas"></canvas>
            <br><br>
            <button id="startBtn">Start Camera</button>
            <button id="stopBtn" disabled>Stop Camera</button>
        </div>
        
        <div class="controls-section">
            <h2>🎛️ Controls</h2>
            
            <div class="control-group">
                <label>Detection Categories:</label>
                <div>
                    <input type="checkbox" id="phone" value="phone" checked> Phone Numbers<br>
                    <input type="checkbox" id="credit_card" value="credit_card" checked> Credit Cards<br>
                    <input type="checkbox" id="email" value="email" checked> Email Addresses<br>
                    <input type="checkbox" id="address" value="address" checked> Addresses<br>
                    <input type="checkbox" id="api_key" value="api_key" checked> API Keys<br>
                </div>
            </div>
            
            <div class="control-group">
                <label for="redactionMethod">Redaction Method:</label>
                <select id="redactionMethod">
                    <option value="gaussian">Gaussian Blur</option>
                    <option value="pixelate">Pixelate</option>
                    <option value="solid">Solid Color</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="detectorStride">Detector Stride:</label>
                <input type="range" id="detectorStride" min="1" max="10" value="3">
                <span id="detectorStrideValue">3</span>
            </div>
            
            <h2>📊 Performance</h2>
            <div class="stats">
                <div class="metric">
                    <span>FPS:</span>
                    <span id="fps">0</span>
                </div>
                <div class="metric">
                    <span>Latency:</span>
                    <span id="latency">0 ms</span>
                </div>
                <div class="metric">
                    <span>Processing Time:</span>
                    <span id="processingTime">0 ms</span>
                </div>
            </div>
            
            <h2>🔍 Detection Stats</h2>
            <div class="stats">
                <div class="metric">
                    <span>Phone Numbers:</span>
                    <span id="phoneCount">0</span>
                </div>
                <div class="metric">
                    <span>Email Addresses:</span>
                    <span id="emailCount">0</span>
                </div>
                <div class="metric">
                    <span>Credit Cards:</span>
                    <span id="cardCount">0</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let localVideo = document.getElementById('localVideo');
        let processedCanvas = document.getElementById('processedCanvas');
        let ctx = processedCanvas.getContext('2d');
        let mediaStream = null;
        let websocket = null;
        let isProcessing = false;
        
        // WebSocket connection
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            websocket = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            websocket.onopen = function(event) {
                console.log('WebSocket connected');
            };
            
            websocket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'processed_frame') {
                    // Display processed frame
                    const img = new Image();
                    img.onload = function() {
                        processedCanvas.width = img.width;
                        processedCanvas.height = img.height;
                        ctx.drawImage(img, 0, 0);
                    };
                    img.src = 'data:image/jpeg;base64,' + data.frame;
                } else if (data.type === 'stats') {
                    // Update performance stats
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('latency').textContent = data.latency.toFixed(1) + ' ms';
                    document.getElementById('processingTime').textContent = data.processing_time.toFixed(1) + ' ms';
                } else if (data.type === 'detections') {
                    // Update detection counters
                    document.getElementById('phoneCount').textContent = data.phone || 0;
                    document.getElementById('emailCount').textContent = data.email || 0;
                    document.getElementById('cardCount').textContent = data.credit_card || 0;
                }
            };
            
            websocket.onclose = function(event) {
                console.log('WebSocket disconnected');
                setTimeout(connectWebSocket, 1000); // Reconnect after 1 second
            };
        }
        
        // Start camera
        document.getElementById('startBtn').onclick = async function() {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 }
                });
                localVideo.srcObject = mediaStream;
                
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                
                connectWebSocket();
                startFrameCapture();
            } catch (err) {
                console.error('Error accessing camera:', err);
            }
        };
        
        // Stop camera
        document.getElementById('stopBtn').onclick = function() {
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
            
            if (websocket) {
                websocket.close();
            }
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        };
        
        // Capture and send frames
        function startFrameCapture() {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            function captureFrame() {
                if (!mediaStream || !websocket || websocket.readyState !== WebSocket.OPEN) {
                    return;
                }
                
                canvas.width = localVideo.videoWidth;
                canvas.height = localVideo.videoHeight;
                ctx.drawImage(localVideo, 0, 0);
                
                canvas.toBlob(function(blob) {
                    if (blob && !isProcessing) {
                        isProcessing = true;
                        const reader = new FileReader();
                        reader.onload = function() {
                            const base64 = reader.result.split(',')[1];
                            websocket.send(JSON.stringify({
                                type: 'frame',
                                data: base64
                            }));
                            isProcessing = false;
                        };
                        reader.readAsDataURL(blob);
                    }
                }, 'image/jpeg', 0.8);
                
                setTimeout(captureFrame, 100); // 10 FPS
            }
            
            captureFrame();
        }
        
        // Control handlers
        document.getElementById('detectorStride').oninput = function() {
            document.getElementById('detectorStrideValue').textContent = this.value;
            updateConfig();
        };
        
        document.getElementById('redactionMethod').onchange = updateConfig;
        
        document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.onchange = updateConfig;
        });
        
        function updateConfig() {
            if (!websocket || websocket.readyState !== WebSocket.OPEN) return;
            
            const categories = Array.from(document.querySelectorAll('input[type="checkbox"]:checked'))
                .map(cb => cb.value);
            
            const config = {
                type: 'config_update',
                categories: categories,
                redaction_method: document.getElementById('redactionMethod').value,
                detector_stride: parseInt(document.getElementById('detectorStride').value)
            };
            
            websocket.send(JSON.stringify(config));
        }
    </script>
</body>
</html>
        """
        
        with open(static_dir / "index.html", "w") as f:
            f.write(html_content)
        
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def read_root():
            """Serve the main HTML page."""
            with open("static/index.html", "r") as f:
                return HTMLResponse(content=f.read())
        
        @self.app.get("/config")
        async def get_config():
            """Get current configuration."""
            if not self.config:
                raise HTTPException(status_code=500, detail="Configuration not loaded")
            
            return {
                "categories": self.config.classification.categories,
                "redaction_method": self.config.redaction.default_method,
                "detector_stride": self.config.realtime.detector_stride,
                "ocr_refresh_stride": self.config.realtime.ocr_refresh_stride
            }
        
        @self.app.post("/config")
        async def update_config(config_update: ConfigUpdate):
            """Update configuration."""
            if not self.config:
                raise HTTPException(status_code=500, detail="Configuration not loaded")
            
            if config_update.categories is not None:
                self.config.classification.categories = config_update.categories
            
            if config_update.redaction_method is not None:
                self.config.redaction.default_method = config_update.redaction_method
            
            if config_update.detector_stride is not None:
                self.config.realtime.detector_stride = config_update.detector_stride
            
            if config_update.ocr_refresh_stride is not None:
                self.config.realtime.ocr_refresh_stride = config_update.ocr_refresh_stride
            
            return {"status": "updated"}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time video processing."""
            await websocket.accept()
            connection_id = str(uuid.uuid4())
            self.active_connections.add(websocket)
            
            # Create pipeline for this connection
            if self.config:
                pipeline = RealtimePipeline(self.config)
                pipeline.start()
                self.connection_pipelines[connection_id] = pipeline
            
            try:
                while True:
                    # Receive message from client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message["type"] == "frame":
                        # Process video frame
                        await self._process_frame(websocket, connection_id, message["data"])
                    
                    elif message["type"] == "config_update":
                        # Update configuration for this connection
                        await self._update_connection_config(connection_id, message)
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket {connection_id} disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                # Cleanup
                self.active_connections.discard(websocket)
                if connection_id in self.connection_pipelines:
                    self.connection_pipelines[connection_id].stop()
                    del self.connection_pipelines[connection_id]
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "pipeline_active": self.pipeline is not None}
    
    async def _process_frame(self, websocket: WebSocket, connection_id: str, frame_data: str):
        """Process a video frame and send results back."""
        try:
            # Decode base64 frame
            import base64
            frame_bytes = base64.b64decode(frame_data)
            
            # Convert to numpy array
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return
            
            # Get pipeline for this connection
            pipeline = self.connection_pipelines.get(connection_id)
            if not pipeline:
                return
            
            # Process frame
            processed_frame = pipeline.process_frame(frame, 0)
            
            # Encode processed frame
            _, buffer = cv2.imencode('.jpg', processed_frame)
            processed_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send processed frame
            await websocket.send_text(json.dumps({
                "type": "processed_frame",
                "frame": processed_base64
            }))
            
            # Send performance stats
            stats = pipeline.get_stats()
            await websocket.send_text(json.dumps({
                "type": "stats",
                "fps": 30.0,  # Placeholder
                "latency": 50.0,  # Placeholder
                "processing_time": 25.0  # Placeholder
            }))
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    async def _update_connection_config(self, connection_id: str, config_data: dict):
        """Update configuration for a specific connection."""
        pipeline = self.connection_pipelines.get(connection_id)
        if not pipeline:
            return
        
        # Update pipeline configuration
        if "categories" in config_data:
            pipeline.config.classification.categories = config_data["categories"]
        
        if "redaction_method" in config_data:
            pipeline.config.redaction.default_method = config_data["redaction_method"]
        
        if "detector_stride" in config_data:
            pipeline.config.realtime.detector_stride = config_data["detector_stride"]
    
    def get_app(self):
        """Get the FastAPI application instance."""
        return self.app


def create_app():
    """Create and return FastAPI application."""
    app_instance = FastAPIApp()
    return app_instance.get_app()


if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )