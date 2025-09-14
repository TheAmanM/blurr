import pytest
import asyncio
from fastapi.testclient import TestClient
from main import app
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import VideoFrame
import numpy as np
import cv2

client = TestClient(app)

# Dummy video track for testing (sends blank frames)
class DummyVideoTrack(MediaStreamTrack):
    kind = "video"

    async def recv(self):
        # Create a simple black frame (480x640)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = 0
        frame.time_base = 1 / 30
        return frame


@pytest.mark.asyncio
async def test_offer_with_real_pc():
    pc = RTCPeerConnection()
    pc.addTrack(DummyVideoTrack())

    # Create offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # Send offer to backend
    response = client.post("/offer", json={
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })
    assert response.status_code == 200

    # Parse backend answer
    answer = response.json()
    remote_desc = RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
    await pc.setRemoteDescription(remote_desc)

    # Assert that remote description was accepted
    assert pc.remoteDescription is not None
    assert pc.remoteDescription.type == "answer"

