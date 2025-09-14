import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing - you can restrict this later
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track wrapper that applies blur
class BlurredTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # Blur the entire frame
        blurred = cv2.GaussianBlur(img, (51, 51), 0)

        # Return new VideoFrame
        new_frame = VideoFrame.from_ndarray(blurred, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


@app.post("/offer")
async def offer(request: Request):
    """Accepts a WebRTC offer and returns an answer with blurred video."""
    data = await request.json()
    offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

    pc = RTCPeerConnection()

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(BlurredTrack(track))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }
