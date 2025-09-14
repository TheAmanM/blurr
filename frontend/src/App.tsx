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

      // Handle incoming answers (this part is fine)
      if (data?.answer && pc.current.signalingState === "have-local-offer") {
        const answerDescription = new RTCSessionDescription(data.answer);
        await pc.current.setRemoteDescription(answerDescription);
      }

      // Handle incoming offers
      if (data?.offer) {
        // --- THIS IS THE CRITICAL FIX ---
        // The creator should ignore offers until the connection is established.
        if (isOfferer && pc.current.connectionState !== "connected") {
          return;
        }

        // Prevent processing the same offer again
        if (pc.current.currentRemoteDescription?.sdp === data.offer.sdp) {
          return;
        }

        const offerDescription = new RTCSessionDescription(data.offer);
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

    // 1. Create a document reference locally without saving to Firestore yet.
    // This gives us an ID to work with immediately.
    const callDocRef = doc(collection(db, "calls"));
    const newCallId = callDocRef.id;

    // 2. Initialize the peer connection with the new ID.
    initializePeerConnection(newCallId, true);
    if (!pc.current) return console.error("Peer connection not created");

    // 3. Set up all necessary listeners.
    const offerCandidatesCol = collection(callDocRef, "offerCandidates");
    const answerCandidatesCol = collection(callDocRef, "answerCandidates");

    pc.current.onicecandidate = (event) => {
      if (event.candidate) addDoc(offerCandidatesCol, event.candidate.toJSON());
    };

    setupSignalingListeners(callDocRef, true);

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

    // 4. Create the offer.
    const offerDescription = await pc.current.createOffer();
    await pc.current.setLocalDescription(offerDescription);
    const offer = { sdp: offerDescription.sdp, type: offerDescription.type };

    // 5. Now, create the document in Firestore WITH the offer included.
    await setDoc(callDocRef, { offer });

    // 6. FINALLY, update the UI state to show the user the Call ID.
    setCallId(newCallId);
    setCallStatus("waiting");
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
            {isMuted ? (
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                ></path>
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M5 5l14 14"
                ></path>
              </svg>
            ) : (
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                ></path>
              </svg>
            )}
          </button>
          <button
            onClick={toggleVideo}
            className={`p-4 rounded-full transition-colors ${
              isVideoOff ? "bg-red-600" : "bg-gray-600 hover:bg-gray-500"
            }`}
          >
            {isVideoOff ? (
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                ></path>
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M1 1l22 22"
                ></path>
              </svg>
            ) : (
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                ></path>
              </svg>
            )}
          </button>
          <button
            onClick={toggleScreenShare}
            className="p-4 rounded-full transition-colors bg-gray-600 hover:bg-gray-500"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
              ></path>
            </svg>
          </button>
          <button
            onClick={() => setShowEndCallConfirm(true)}
            className="p-4 rounded-full bg-red-600 hover:bg-red-700"
          >
            <svg
              className="w-6 h-6"
              fill="currentColor"
              viewBox="0 -256 1792 1792"
            >
              <g transform="matrix(1,0,0,-1,159.45763,1293.0169)">
                <path
                  d="m 1408,296 q 0,-27 -10,-70.5 Q 1388,182 1377,157 1356,107 1255,51 1161,0 1069,0 1042,0 1016.5,3.5 991,7 959,16 927,25 911.5,30.5 896,36 856,51 816,66 807,69 709,104 632,152 504,231 367.5,367.5 231,504 152,632 104,709 69,807 66,816 51,856 36,896 30.5,911.5 25,927 16,959 7,991 3.5,1016.5 0,1042 0,1069 q 0,92 51,186 56,101 106,122 25,11 68.5,21 43.5,10 70.5,10 14,0 21,-3 18,-6 53,-76 11,-19 30,-54 19,-35 35,-63.5 16,-28.5 31,-53.5 3,-4 17.5,-25 14.5,-21 21.5,-35.5 7,-14.5 7,-28.5 0,-20 -28.5,-50 -28.5,-30 -62,-55 -33.5,-25 -62,-53 -28.5,-28 -28.5,-46 0,-9 5,-22.5 5,-13.5 8.5,-20.5 3.5,-7 14,-24 10.5,-17 11.5,-19 76,-137 174,-235 98,-98 235,-174 2,-1 19,-11.5 17,-10.5 24,-14 7,-3.5 20.5,-8.5 13.5,-5 22.5,-5 18,0 46,28.5 28,28.5 53,62 25,33.5 55,62 30,28.5 50,28.5 14,0 28.5,-7 14.5,-7 35.5,-21.5 21,-14.5 25,-17.5 25,-15 53.5,-31 28.5,-16 63.5,-35 35,-19 54,-30 70,-35 76,-53 3,-7 3,-21 z"
                  style={{ fill: "currentColor" }}
                />
              </g>
            </svg>
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
