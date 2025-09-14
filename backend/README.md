# WebRTC Blur Demo Backend

This backend is a FastAPI server that receives a WebRTC video stream, applies a Gaussian blur to each frame using OpenCV, and returns the processed (blurred) video stream back to the client.

## Features

- Accepts WebRTC video streams via `/offer` endpoint.
- Currently applies a 51x51 Gaussian blur to every video frame server-side.
- Returns the blurred video stream to the client in real time.
- CORS enabled for all origins (for development/testing).
- Designed to work with the provided frontend (`sample_endpoint.html`).

## Endpoints

- `POST /offer`: Accepts a WebRTC offer (SDP), negotiates a connection, and returns an answer (SDP). The returned stream is the blurred version of the input video.

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies:
  - fastapi
  - uvicorn
  - aiortc
  - opencv-python
  - av
  - pytest, pytest-asyncio, httpx (for testing)

## Running the Backend

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Start the FastAPI server:**
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. **Test the endpoint:**
   - Open `sample_endpoint.html` in your browser.
   - Click "Start Camera" to begin streaming and see the blurred video returned from the backend.

## Testing

- Before running tests make sure server is active. Run tests with:
  ```
  pytest
  ```

## Notes

- The backend currently allows all CORS origins for easy local development.
- Only video streams are processed; audio is ignored.
- The Gaussian blur kernel size is fixed at 51x51 for demonstration.
- Future steps include adding custom filtering logic to only filter information
  that should be private.

---
