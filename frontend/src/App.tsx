import React, { useState, useRef, useEffect } from "react";
import { initializeApp } from "firebase/app";
import {
  getFirestore,
  doc,
  collection,
  addDoc,
  setDoc,
  getDoc,
  onSnapshot,
  updateDoc,
  deleteField,
  writeBatch,
  query,
} from "firebase/firestore";

// --- Firebase Configuration ---
// NOTE: Replace with your actual Firebase config
const firebaseConfig = {
  apiKey: "AIzaSyB9cXojIKf2SsXjb69UedLHPipy7x6f-S8",
  authDomain: "web-rtc-demo-95be5.firebaseapp.com",
  projectId: "web-rtc-demo-95be5",
  storageBucket: "web-rtc-demo-95be5.firebasestorage.app",
  messagingSenderId: "61059022514",
  appId: "1:61059022514:web:6b338e9a51166ec0d793ab",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

// --- STUN/TURN servers for NAT traversal ---
const servers = {
  iceServers: [
    {
      urls: ["stun:stun1.l.google.com:19302", "stun:stun2.l.google.com:19302"],
    },
    // Add a TURN server here for better connectivity
  ],
  iceCandidatePoolSize: 10,
};

function App() {
  // State variables for media streams (simplified)
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [isSharingScreen, setIsSharingScreen] = useState(false);

  // State for call logic and UI
  const [isMuted, setIsMuted] = useState(false);
  const [isVideoOff, setIsVideoOff] = useState(false);
  const [showEndCallConfirm, setShowEndCallConfirm] = useState(false);
  const [callId, setCallId] = useState("");
  const [joiningCallId, setJoiningCallId] = useState("");
  const [callStatus, setCallStatus] = useState<
    "idle" | "creating" | "waiting" | "connected"
  >("idle");

  // Refs for non-re-rendering objects
  const pc = useRef<RTCPeerConnection | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const videoSenderRef = useRef<RTCRtpSender | null>(null); // Replaces screenSenderRef
  const signalingUnsubscribers = useRef<(() => void)[]>([]);

  // Refs for video elements (simplified)
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);

  /**
   * Initializes the webcam and microphone stream.
   */
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      localStreamRef.current = stream;
      setLocalStream(stream);
    } catch (error) {
      console.error("Error accessing media devices:", error);
      alert("Could not access webcam. Please check permissions.");
    }
  };

  /**
   * Sets up a universal listener for signaling messages (offers/answers) from Firestore.
   */
  const setupSignalingListeners = (callDocRef: any, isOfferer: boolean) => {
    const mainUnsubscriber = onSnapshot(callDocRef, async (snapshot: any) => {
      const data = snapshot.data();
      if (!pc.current) return;

      if (data?.answer && pc.current.signalingState === "have-local-offer") {
        const answerDescription = new RTCSessionDescription(data.answer);
        await pc.current.setRemoteDescription(answerDescription);
      }

      if (data?.offer) {
        const shouldProcessOffer =
          isOfferer || pc.current.connectionState === "connected";

        if (shouldProcessOffer) {
          const offerDescription = new RTCSessionDescription(data.offer);
          if (
            pc.current.currentRemoteDescription?.sdp !== offerDescription.sdp
          ) {
            await pc.current.setRemoteDescription(offerDescription);
            const answer = await pc.current.createAnswer();
            await pc.current.setLocalDescription(answer);
            if (pc.current.localDescription) {
              await updateDoc(callDocRef, {
                answer: pc.current.localDescription.toJSON(),
                offer: deleteField(),
              });
            }
          }
        }
      }
    });
    signalingUnsubscribers.current.push(mainUnsubscriber);
  };

  /**
   * Initializes the RTCPeerConnection object for a new call.
   */
  const initializePeerConnection = (
    currentCallId: string,
    isOfferer: boolean
  ) => {
    const newPc = new RTCPeerConnection(servers);

    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => {
        const sender = newPc.addTrack(track, localStreamRef.current!);
        if (track.kind === "video") {
          videoSenderRef.current = sender; // Capture the video sender
        }
      });
    }

    newPc.ontrack = (event) => {
      setRemoteStream((prevStream) => {
        const newStream = prevStream
          ? new MediaStream(prevStream.getTracks())
          : new MediaStream();
        if (!newStream.getTrackById(event.track.id)) {
          newStream.addTrack(event.track);
        }
        return newStream;
      });
    };

    newPc.onconnectionstatechange = () => {
      if (newPc.connectionState === "connected") {
        setCallStatus("connected");
      }
    };

    newPc.onnegotiationneeded = async () => {
      if (!isOfferer && newPc.connectionState !== "connected") {
        return;
      }
      if (newPc.signalingState !== "stable" || !currentCallId) {
        return;
      }
      try {
        const offer = await newPc.createOffer();
        await newPc.setLocalDescription(offer);
        if (newPc.localDescription) {
          const callDocRef = doc(db, "calls", currentCallId);
          await updateDoc(callDocRef, {
            offer: newPc.localDescription.toJSON(),
          });
        }
      } catch (error) {
        console.error("Error during negotiation:", error);
      }
    };

    pc.current = newPc;
  };

  /**
   * Creates a new call, generates an offer, and writes it to Firestore.
   */
  const handleCreateCall = async () => {
    if (!localStream) return alert("Please start your webcam first!");
    setCallStatus("creating");
    const callDocRef = await addDoc(collection(db, "calls"), {});
    const newCallId = callDocRef.id;
    setCallId(newCallId);

    initializePeerConnection(newCallId, true);
    if (!pc.current) return console.error("Peer connection not created");

    const offerCandidatesCol = collection(callDocRef, "offerCandidates");
    const answerCandidatesCol = collection(callDocRef, "answerCandidates");

    pc.current.onicecandidate = (event) => {
      if (event.candidate) addDoc(offerCandidatesCol, event.candidate.toJSON());
    };

    setupSignalingListeners(callDocRef, true);

    const offerDescription = await pc.current.createOffer();
    await pc.current.setLocalDescription(offerDescription);
    const offer = { sdp: offerDescription.sdp, type: offerDescription.type };
    await setDoc(callDocRef, { offer });

    setCallStatus("waiting");

    const unsubCandidates = onSnapshot(
      query(answerCandidatesCol),
      (snapshot) => {
        snapshot.docChanges().forEach((change) => {
          if (change.type === "added") {
            pc.current?.addIceCandidate(new RTCIceCandidate(change.doc.data()));
          }
        });
      }
    );
    signalingUnsubscribers.current.push(unsubCandidates);
  };

  /**
   * Joins an existing call by reading the offer and creating an answer.
   */
  const handleJoinCall = async () => {
    if (!localStream) return alert("Please start your webcam first!");
    if (!joiningCallId) return alert("Please enter a Call ID to join.");

    setCallId(joiningCallId);
    setCallStatus("creating");
    const callDocRef = doc(db, "calls", joiningCallId);
    const callDocSnap = await getDoc(callDocRef);
    if (!callDocSnap.exists()) return alert("Call ID not found.");

    initializePeerConnection(joiningCallId, false);
    if (!pc.current) return console.error("Peer connection not created");

    const offerCandidatesCol = collection(callDocRef, "offerCandidates");
    const answerCandidatesCol = collection(callDocRef, "answerCandidates");

    pc.current.onicecandidate = (event) => {
      if (event.candidate)
        addDoc(answerCandidatesCol, event.candidate.toJSON());
    };

    setupSignalingListeners(callDocRef, false);

    const offerDescription = callDocSnap.data().offer;
    await pc.current.setRemoteDescription(
      new RTCSessionDescription(offerDescription)
    );
    const answerDescription = await pc.current.createAnswer();
    await pc.current.setLocalDescription(answerDescription);
    const answer = { type: answerDescription.type, sdp: answerDescription.sdp };
    await updateDoc(callDocRef, { answer, offer: deleteField() });

    const unsubCandidates = onSnapshot(
      query(offerCandidatesCol),
      (snapshot) => {
        snapshot.docChanges().forEach((change) => {
          if (change.type === "added") {
            pc.current?.addIceCandidate(new RTCIceCandidate(change.doc.data()));
          }
        });
      }
    );
    signalingUnsubscribers.current.push(unsubCandidates);
  };

  const hangUp = async () => {
    pc.current?.close();
    if (callId) {
      const callDocRef = doc(db, "calls", callId);
      const batch = writeBatch(db);
      batch.delete(callDocRef);
      await batch.commit();
    }

    signalingUnsubscribers.current.forEach((unsub) => unsub());
    signalingUnsubscribers.current = [];

    localStreamRef.current?.getTracks().forEach((track) => track.stop());
    setLocalStream(null);
    setRemoteStream(null);
    setCallId("");
    setJoiningCallId("");
    setCallStatus("idle");
    setShowEndCallConfirm(false);
    setIsMuted(false);
    setIsVideoOff(false);
    setIsSharingScreen(false);
    localStreamRef.current = null;
    videoSenderRef.current = null;
    pc.current = null;
  };

  const toggleMic = () => {
    if (localStreamRef.current) {
      localStreamRef.current.getAudioTracks().forEach((track) => {
        track.enabled = !track.enabled;
      });
      setIsMuted((prev) => !prev);
    }
  };

  const toggleVideo = () => {
    if (localStream) {
      localStream.getVideoTracks().forEach((track) => {
        track.enabled = !track.enabled;
      });
      setIsVideoOff((prev) => !prev);
    }
  };

  const toggleScreenShare = async () => {
    if (!pc.current || !videoSenderRef.current) return;

    if (isSharingScreen) {
      if (!localStreamRef.current) return;
      const cameraTrack = localStreamRef.current.getVideoTracks()[0];
      await videoSenderRef.current.replaceTrack(cameraTrack);
      setLocalStream(localStreamRef.current);
      setIsSharingScreen(false);
    } else {
      try {
        const screenStream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
        });
        const screenTrack = screenStream.getVideoTracks()[0];

        await videoSenderRef.current.replaceTrack(screenTrack);

        const audioTrack = localStreamRef.current?.getAudioTracks()[0];
        const newLocalStream = new MediaStream([
          screenTrack,
          ...(audioTrack ? [audioTrack] : []),
        ]);
        setLocalStream(newLocalStream);
        setIsSharingScreen(true);

        screenTrack.onended = () => {
          if (videoSenderRef.current && localStreamRef.current) {
            const cameraTrack = localStreamRef.current.getVideoTracks()[0];
            videoSenderRef.current.replaceTrack(cameraTrack);
            setLocalStream(localStreamRef.current);
            setIsSharingScreen(false);
          }
        };
      } catch (error) {
        console.error("Error starting screen share:", error);
        setIsSharingScreen(false);
      }
    }
  };

  useEffect(() => {
    if (localVideoRef.current && localStream)
      localVideoRef.current.srcObject = localStream;
  }, [localStream]);

  useEffect(() => {
    if (remoteVideoRef.current && remoteStream)
      remoteVideoRef.current.srcObject = remoteStream;
  }, [remoteStream]);

  const copyToClipboard = (text: string) => {
    if (!text) return;
    navigator.clipboard
      .writeText(text)
      .then(() => alert("Copied to clipboard!"));
  };

  const renderVideos = () => {
    return (
      <div className="flex-grow grid md:grid-cols-2 gap-4 h-full place-items-center">
        {/* Local Video */}
        <div className="relative bg-black rounded-lg overflow-hidden shadow-lg w-full aspect-video">
          {localStream ? (
            <video
              ref={localVideoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-contain"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <p className="text-gray-400">Start your webcam to begin</p>
            </div>
          )}
          <span className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white text-sm px-2 py-1 rounded">
            {isSharingScreen ? "Your Screen" : "You"}
          </span>
        </div>

        {/* Remote Video */}
        <div className="relative bg-black rounded-lg overflow-hidden shadow-lg w-full aspect-video">
          {remoteStream ? (
            <video
              ref={remoteVideoRef}
              autoPlay
              playsInline
              className="w-full h-full object-contain"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <p className="text-gray-400">Waiting for friend to connect...</p>
            </div>
          )}
          <span className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white text-sm px-2 py-1 rounded">
            Friend
          </span>
        </div>
      </div>
    );
  };

  const renderUiByStatus = () => {
    if (callStatus === "connected") {
      return (
        <div className="flex justify-center items-center gap-4 h-full">
          <button
            onClick={toggleMic}
            className={`p-4 rounded-full transition-colors ${
              isMuted ? "bg-red-600" : "bg-gray-600 hover:bg-gray-500"
            }`}
          >
            {/* Mic SVG */}
          </button>
          <button
            onClick={toggleVideo}
            className={`p-4 rounded-full transition-colors ${
              isVideoOff ? "bg-red-600" : "bg-gray-600 hover:bg-gray-500"
            }`}
          >
            {/* Video SVG */}
          </button>
          <button
            onClick={toggleScreenShare}
            className={`p-4 rounded-full transition-colors ${
              isSharingScreen ? "bg-blue-500" : "bg-gray-600 hover:bg-gray-500"
            }`}
          >
            {/* Screen Share SVG */}
          </button>
          <button
            onClick={() => setShowEndCallConfirm(true)}
            className="p-4 rounded-full bg-red-600 hover:bg-red-700"
          >
            {/* Hang Up SVG */}
          </button>
        </div>
      );
    }
    if (callStatus === "waiting") {
      return (
        <div className="flex flex-col items-center justify-center text-center">
          <h2 className="text-2xl font-bold text-teal-300 mb-4">
            Call Created!
          </h2>
          <p className="mb-4">Share this Call ID with your friend:</p>
          <div className="bg-gray-900 p-3 rounded-lg flex items-center gap-4">
            <span className="font-mono text-lg">{callId}</span>
            <button
              onClick={() => copyToClipboard(callId)}
              className="p-2 bg-gray-600 hover:bg-gray-500 rounded"
            >
              Copy ID
            </button>
          </div>
          <button
            onClick={hangUp}
            className="mt-8 px-8 py-3 bg-red-600 hover:bg-red-700 rounded-lg font-semibold"
          >
            Cancel
          </button>
        </div>
      );
    }
    return (
      <div className="text-center">
        <button
          onClick={startWebcam}
          disabled={!!localStream}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg font-semibold transition-colors duration-300 mb-4 w-full md:w-auto"
        >
          1. Start Webcam
        </button>
        <div className="flex flex-col md:flex-row justify-center gap-4">
          <button
            onClick={handleCreateCall}
            disabled={!localStream}
            className="px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-lg font-semibold"
          >
            2. Create Call
          </button>
          <div className="flex items-center gap-2">
            <input
              value={joiningCallId}
              onChange={(e) => setJoiningCallId(e.target.value)}
              placeholder="Enter Call ID"
              className="bg-gray-900 p-3 rounded-lg font-mono text-center w-full"
            />
            <button
              onClick={handleJoinCall}
              disabled={!localStream || !joiningCallId}
              className="px-6 py-3 bg-teal-600 hover:bg-teal-700 disabled:bg-gray-600 rounded-lg font-semibold"
            >
              Join
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-gray-900 text-white min-h-screen flex flex-col items-center p-4 font-sans">
      <div
        className="w-full max-w-6xl mx-auto flex flex-col"
        style={{ height: "95vh" }}
      >
        <header className="text-center mb-4 flex-shrink-0">
          <h1 className="text-4xl font-bold text-teal-400">
            Serverless WebRTC
          </h1>
          <p className="text-gray-400">
            Video calling with in-place screen sharing!
          </p>
        </header>
        <div className="flex-grow min-h-0">{renderVideos()}</div>
        <main className="flex-shrink-0 bg-gray-800 p-6 rounded-lg shadow-2xl mt-4">
          {renderUiByStatus()}
        </main>
      </div>
      {showEndCallConfirm && (
        <div className="absolute inset-0 bg-black bg-opacity-70 flex justify-center items-center z-10">
          <div className="bg-gray-800 p-8 rounded-lg shadow-xl text-center">
            <h2 className="text-2xl mb-4 font-bold">End Call?</h2>
            <p className="text-gray-400 mb-6">
              Are you sure you want to end the call?
            </p>
            <div className="flex justify-center gap-4">
              <button
                onClick={() => setShowEndCallConfirm(false)}
                className="px-6 py-2 bg-gray-600 hover:bg-gray-500 rounded-lg font-semibold"
              >
                Cancel
              </button>
              <button
                onClick={hangUp}
                className="px-6 py-2 bg-red-600 hover:bg-red-700 rounded-lg font-semibold"
              >
                End Call
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
