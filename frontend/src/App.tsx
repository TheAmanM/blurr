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

// --- STUN servers for NAT traversal ---
const servers = {
  iceServers: [
    {
      urls: ["stun:stun1.l.google.com:19302", "stun:stun2.l.google.com:19302"],
    },
  ],
  iceCandidatePoolSize: 10,
};

function App() {
  // State variables for media streams
  const [localCamStream, setLocalCamStream] = useState<MediaStream | null>(
    null
  );
  const [localScreenStream, setLocalScreenStream] =
    useState<MediaStream | null>(null);
  const [remoteCamStream, setRemoteCamStream] = useState<MediaStream | null>(
    null
  );
  const [remoteScreenStream, setRemoteScreenStream] =
    useState<MediaStream | null>(null);

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
  const localCamStreamRef = useRef<MediaStream | null>(null);
  const screenSenderRef = useRef<RTCRtpSender | null>(null);
  const signalingUnsubscribers = useRef<(() => void)[]>([]);

  // Refs for video elements
  const localCamVideoRef = useRef<HTMLVideoElement>(null);
  const localScreenVideoRef = useRef<HTMLVideoElement>(null);
  const remoteCamVideoRef = useRef<HTMLVideoElement>(null);
  const remoteScreenVideoRef = useRef<HTMLVideoElement>(null);

  /**
   * Initializes the webcam and microphone stream.
   */
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      localCamStreamRef.current = stream;
      setLocalCamStream(stream);
    } catch (error) {
      console.error("Error accessing media devices:", error);
      alert("Could not access webcam. Please check permissions.");
    }
  };

  /**
   * Sets up a universal listener for signaling messages (offers/answers) from Firestore.
   */
  const setupSignalingListeners = (callDocRef: any) => {
    const mainUnsubscriber = onSnapshot(callDocRef, async (snapshot: any) => {
      const data = snapshot.data();
      if (!pc.current) return;

      // Handle incoming answers. An answer is a response to an offer we sent.
      if (data?.answer && pc.current.signalingState === "have-local-offer") {
        console.log("Received answer, setting remote description.");
        const answerDescription = new RTCSessionDescription(data.answer);
        await pc.current.setRemoteDescription(answerDescription);
      }

      // Handle incoming offers. An offer is an intention to start/change a call.
      if (data?.offer && pc.current.signalingState === "stable") {
        console.log("[DEBUG] Received a new offer over Firebase."); // <-- ADD THIS
        if (data.offer.sdp === pc.current.localDescription?.sdp) {
          return;
        }

        const offerDescription = new RTCSessionDescription(data.offer);
        const isNewOffer =
          !pc.current.currentRemoteDescription ||
          pc.current.currentRemoteDescription.sdp !== offerDescription.sdp;

        if (isNewOffer) {
          console.log("Received new offer, creating answer.");
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
    });
    signalingUnsubscribers.current.push(mainUnsubscriber);
  };

  /**
   * Initializes the RTCPeerConnection object for a new call.
   */
  const initializePeerConnection = () => {
    const newPc = new RTCPeerConnection(servers);

    // Add local camera and mic tracks to the connection
    if (localCamStreamRef.current) {
      localCamStreamRef.current.getTracks().forEach((track) => {
        newPc.addTrack(track, localCamStreamRef.current!);
      });
    }

    // This is the updated ontrack handler
    newPc.ontrack = (event) => {
      console.log(
        "[DEBUG] ontrack event fired! Received remote track:",
        event.track.kind,
        event.track.id
      ); // <-- MODIFY THIS
      const track = event.track;

      // Differentiate between a screen share track and a camera/mic track
      const isScreenTrack =
        track.kind === "video" && track.getSettings().displaySurface;

      if (isScreenTrack) {
        // Handle the remote screen share stream
        setRemoteScreenStream(new MediaStream([track]));
      } else {
        // Handle remote camera and microphone tracks by adding them to a single stream
        setRemoteCamStream((prevStream) => {
          // If a stream already exists, we'll add to it. Otherwise, create a new one.
          const newStream = prevStream
            ? new MediaStream(prevStream.getTracks())
            : new MediaStream();

          // Add the new track if it's not already in the stream
          if (!newStream.getTrackById(track.id)) {
            newStream.addTrack(track);
          }

          return newStream;
        });
      }
    };

    newPc.onconnectionstatechange = () => {
      console.log("Connection state:", newPc.connectionState);
      if (newPc.connectionState === "connected") {
        setCallStatus("connected");
      }
    };

    newPc.onnegotiationneeded = async () => {
      console.log(
        `[DEBUG] onnegotiationneeded fired! Signaling state: ${newPc.signalingState}, Call ID: ${callId}`
      ); // <-- MODIFY THIS LINE
      if (newPc.signalingState !== "stable" || !callId) {
        console.log("Skipping negotiation:", newPc.signalingState);
        console.log(
          "[DEBUG] Skipping negotiation due to unstable state or missing callId."
        ); // <-- ADD THIS
        return;
      }
      console.log("Negotiation needed, creating new offer.");
      console.log("[DEBUG] Negotiation needed, creating new offer."); // <-- MODIFY THIS
      try {
        const offer = await newPc.createOffer();
        await newPc.setLocalDescription(offer);
        if (newPc.localDescription) {
          const callDocRef = doc(db, "calls", callId);
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
    if (!localCamStream) return alert("Please start your webcam first!");
    setCallStatus("creating");
    const callDocRef = await addDoc(collection(db, "calls"), {});
    const newCallId = callDocRef.id;
    setCallId(newCallId);

    initializePeerConnection();
    if (!pc.current) return console.error("Peer connection not created");

    const offerCandidatesCol = collection(callDocRef, "offerCandidates");
    const answerCandidatesCol = collection(callDocRef, "answerCandidates");

    pc.current.onicecandidate = (event) => {
      if (event.candidate) addDoc(offerCandidatesCol, event.candidate.toJSON());
    };

    setupSignalingListeners(callDocRef);

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
    if (!localCamStream) return alert("Please start your webcam first!");
    if (!joiningCallId) return alert("Please enter a Call ID to join.");

    setCallId(joiningCallId);
    setCallStatus("creating");
    const callDocRef = doc(db, "calls", joiningCallId);
    const callDocSnap = await getDoc(callDocRef);
    if (!callDocSnap.exists()) return alert("Call ID not found.");

    initializePeerConnection();
    if (!pc.current) return console.error("Peer connection not created");

    const offerCandidatesCol = collection(callDocRef, "offerCandidates");
    const answerCandidatesCol = collection(callDocRef, "answerCandidates");

    pc.current.onicecandidate = (event) => {
      if (event.candidate)
        addDoc(answerCandidatesCol, event.candidate.toJSON());
    };

    setupSignalingListeners(callDocRef);

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

    localCamStreamRef.current?.getTracks().forEach((track) => track.stop());
    localScreenStream?.getTracks().forEach((track) => track.stop());
    setLocalCamStream(null);
    setLocalScreenStream(null);
    setRemoteCamStream(null);
    setRemoteScreenStream(null);
    setCallId("");
    setJoiningCallId("");
    setCallStatus("idle");
    setShowEndCallConfirm(false);
    setIsMuted(false);
    setIsVideoOff(false);
    localCamStreamRef.current = null;
    screenSenderRef.current = null;
    pc.current = null;
  };

  const toggleMic = () => {
    if (localCamStreamRef.current) {
      localCamStreamRef.current.getAudioTracks().forEach((track) => {
        track.enabled = !track.enabled;
      });
      setIsMuted((prev) => !prev);
    }
  };

  const toggleVideo = () => {
    if (localCamStreamRef.current) {
      localCamStreamRef.current.getVideoTracks().forEach((track) => {
        track.enabled = !track.enabled;
      });
      setIsVideoOff((prev) => !prev);
    }
  };

  const toggleScreenShare = async () => {
    if (pc.current?.connectionState !== "connected") {
      alert(
        "Please wait for the connection to be fully established before sharing your screen."
      );
      return;
    }
    if (!pc.current) return;
    if (screenSenderRef.current) {
      pc.current.removeTrack(screenSenderRef.current);
      localScreenStream?.getTracks().forEach((track) => track.stop());
      setLocalScreenStream(null);
      screenSenderRef.current = null;
    } else {
      try {
        console.log("[DEBUG] User wants to start screen share."); // <-- ADD THIS
        const screenStream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
        });
        const screenTrack = screenStream.getVideoTracks()[0];
        console.log("[DEBUG] Got screen track:", screenTrack); // <-- ADD THIS
        screenSenderRef.current = pc.current.addTrack(
          screenTrack,
          screenStream
        );
        console.log("[DEBUG] Added screen track to peer connection."); // <-- ADD THIS
        setLocalScreenStream(screenStream);
        screenTrack.onended = () => {
          if (screenSenderRef.current) toggleScreenShare();
        };
      } catch (error) {
        console.error("Error starting screen share:", error);
      }
    }
  };

  useEffect(() => {
    if (localCamVideoRef.current && localCamStream)
      localCamVideoRef.current.srcObject = localCamStream;
  }, [localCamStream]);
  useEffect(() => {
    if (localScreenVideoRef.current && localScreenStream)
      localScreenVideoRef.current.srcObject = localScreenStream;
  }, [localScreenStream]);
  useEffect(() => {
    if (remoteCamVideoRef.current && remoteCamStream)
      remoteCamVideoRef.current.srcObject = remoteCamStream;
  }, [remoteCamStream]);
  useEffect(() => {
    if (remoteScreenVideoRef.current && remoteScreenStream)
      remoteScreenVideoRef.current.srcObject = remoteScreenStream;
  }, [remoteScreenStream]);

  const copyToClipboard = (text: string) => {
    if (!text) return;
    navigator.clipboard
      .writeText(text)
      .then(() => alert("Copied to clipboard!"));
  };

  const renderVideos = () => {
    const localCamVideo = localCamStream && (
      <div
        key="localCam"
        className="relative bg-black rounded-lg overflow-hidden shadow-lg w-full h-full"
      >
        <video
          ref={localCamVideoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-contain"
        />
        <span className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white text-sm px-2 py-1 rounded">
          You
        </span>
      </div>
    );
    const localScreenVideo = localScreenStream && (
      <div
        key="localScreen"
        className="relative bg-black rounded-lg overflow-hidden shadow-lg w-full h-full"
      >
        <video
          ref={localScreenVideoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-contain"
        />
        <span className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white text-sm px-2 py-1 rounded">
          Your Screen
        </span>
      </div>
    );
    const remoteCamVideo = remoteCamStream && (
      <div
        key="remoteCam"
        className="relative bg-black rounded-lg overflow-hidden shadow-lg w-full h-full"
      >
        <video
          ref={remoteCamVideoRef}
          autoPlay
          playsInline
          className="w-full h-full object-contain"
        />
        <span className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white text-sm px-2 py-1 rounded">
          Friend
        </span>
      </div>
    );
    const remoteScreenVideo = remoteScreenStream && (
      <div
        key="remoteScreen"
        className="relative bg-black rounded-lg overflow-hidden shadow-lg w-full h-full"
      >
        <video
          ref={remoteScreenVideoRef}
          autoPlay
          playsInline
          className="w-full h-full object-contain"
        />
        <span className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white text-sm px-2 py-1 rounded">
          Friend's Screen
        </span>
      </div>
    );
    const noRemoteConnectionPlaceholder = !remoteCamStream &&
      !remoteScreenStream && (
        <div
          key="remotePlaceholder"
          className="relative bg-black rounded-lg overflow-hidden shadow-lg w-full h-full flex items-center justify-center"
        >
          <p className="text-gray-400">Waiting for connection...</p>
        </div>
      );
    let mainVideos = [];
    let smallVideos = [];
    if (localScreenVideo) mainVideos.push(localScreenVideo);
    if (remoteScreenVideo) mainVideos.push(remoteScreenVideo);
    if (localCamVideo) smallVideos.push(localCamVideo);
    if (remoteCamVideo) smallVideos.push(remoteCamVideo);
    if (mainVideos.length === 0) {
      mainVideos = smallVideos;
      smallVideos = [];
      if (
        mainVideos.length === 1 &&
        localCamStream &&
        callStatus !== "connected"
      ) {
        if (noRemoteConnectionPlaceholder)
          mainVideos.push(noRemoteConnectionPlaceholder);
      }
    }
    return (
      <div className="flex flex-col h-full">
        <div
          className={`flex-grow grid gap-4 ${
            mainVideos.length > 1 ? "md:grid-cols-2" : "grid-cols-1"
          }`}
        >
          {mainVideos}
        </div>
        {smallVideos.length > 0 && (
          <div className="flex-shrink-0 flex justify-center items-center gap-4 pt-4">
            {smallVideos.map((video) => (
              <div key={video.key} className="w-1/4 max-w-[250px] aspect-video">
                {video}
              </div>
            ))}
          </div>
        )}
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
                d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 S0 116 0v6a3 3 0 01-3 3z"
              ></path>
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M5 5l14 14"
              ></path>
            </svg>
          </button>
          <button
            onClick={toggleVideo}
            className={`p-4 rounded-full transition-colors ${
              isVideoOff ? "bg-red-600" : "bg-gray-600 hover:bg-gray-500"
            }`}
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
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              ></path>
            </svg>
          </button>
          <button
            onClick={toggleScreenShare}
            className={`p-4 rounded-full transition-colors ${
              !!localScreenStream
                ? "bg-blue-500"
                : "bg-gray-600 hover:bg-gray-500"
            }`}
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
          disabled={!!localCamStream}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg font-semibold transition-colors duration-300 mb-4 w-full md:w-auto"
        >
          1. Start Webcam
        </button>
        <div className="flex flex-col md:flex-row justify-center gap-4">
          <button
            onClick={handleCreateCall}
            disabled={!localCamStream}
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
              disabled={!localCamStream || !joiningCallId}
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
            Now with automated signaling via Firebase!
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
