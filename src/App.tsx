import React, { useState, useRef, useEffect } from "react";

// --- STUN servers for NAT traversal (still needed) ---
const servers = {
  iceServers: [
    {
      urls: ["stun:stun1.l.google.com:19302", "stun:stun2.l.google.com:19302"],
    },
  ],
  iceCandidatePoolSize: 10,
};

// --- Helper component for instructional steps ---
const Step = ({
  number,
  title,
  children,
}: {
  number: number;
  title: string;
  children: React.ReactNode;
}) => (
  <div className="bg-gray-700 p-4 rounded-lg">
    <h3 className="text-lg font-semibold text-teal-300 mb-2">
      <span className="bg-teal-500 text-gray-900 rounded-full w-6 h-6 inline-flex items-center justify-center mr-2">
        {number}
      </span>
      {title}
    </h3>
    {children}
  </div>
);

function App() {
  // State variables
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [callMode, setCallMode] = useState<"idle" | "creating" | "joining">(
    "idle"
  );
  const [isMuted, setIsMuted] = useState(false);
  const [isVideoOff, setIsVideoOff] = useState(false);
  const [showEndCallConfirm, setShowEndCallConfirm] = useState(false);
  const [isConnectionEstablished, setIsConnectionEstablished] = useState(false);

  // useRef for mutable objects that don't trigger re-renders
  const pc = useRef<RTCPeerConnection | null>(null);
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);

  // Textarea state for manual signaling
  const [offerData, setOfferData] = useState("");
  const [answerData, setAnswerData] = useState("");

  /**
   * Initializes the webcam and microphone stream.
   */
  const startWebcam = async () => {
    console.log(`[${performance.now().toFixed(2)}ms] Starting webcam...`);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      console.log(
        `[${performance.now().toFixed(2)}ms] Webcam started successfully.`
      );
      setLocalStream(stream);
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error(
        `[${performance.now().toFixed(2)}ms] Error accessing media devices:`,
        error
      );
      alert("Could not access webcam. Please check permissions.");
    }
  };

  /**
   * Initializes the RTCPeerConnection object and sets up listeners.
   */
  const initializePeerConnection = () => {
    console.log(
      `[${performance.now().toFixed(2)}ms] Initializing Peer Connection...`
    );
    const newPc = new RTCPeerConnection(servers);

    // Add local stream tracks to the connection
    if (localStream) {
      console.log(
        `[${performance.now().toFixed(2)}ms] Adding local stream tracks.`
      );
      localStream.getTracks().forEach((track) => {
        newPc.addTrack(track, localStream);
      });
    }

    // Handle incoming remote stream
    newPc.ontrack = (event) => {
      console.log(
        `[${performance.now().toFixed(2)}ms] Remote track received.`,
        event
      );
      setRemoteStream(event.streams[0]);
    };

    // **FIX**: Use connection state to determine when the call is truly established
    newPc.onconnectionstatechange = () => {
      console.log(
        `[${performance.now().toFixed(2)}ms] Connection state changed: ${
          newPc.connectionState
        }`
      );
      if (newPc.connectionState === "connected") {
        console.log(
          `[${performance.now().toFixed(2)}ms] Connection established!`
        );
        setIsConnectionEstablished(true);
      }
    };

    // Collect ICE candidates and combine them into a single string
    let candidates: RTCIceCandidateInit[] = [];
    newPc.onicecandidate = (event) => {
      if (event.candidate) {
        console.log(
          `[${performance.now().toFixed(2)}ms] Found ICE candidate:`,
          event.candidate
        );
        candidates.push(event.candidate.toJSON());
      }
    };

    // When gathering is complete, set the offer/answer data
    newPc.onicegatheringstatechange = () => {
      console.log(
        `[${performance.now().toFixed(2)}ms] ICE gathering state changed: ${
          newPc.iceGatheringState
        }`
      );
      if (newPc.iceGatheringState === "complete") {
        console.log(
          `[${performance.now().toFixed(2)}ms] ICE gathering complete.`
        );
        const sdp = newPc.localDescription;
        if (sdp) {
          const data = { sdp, candidates };
          if (sdp.type === "offer") {
            console.log(
              `[${performance.now().toFixed(2)}ms] Setting offer data.`
            );
            setOfferData(JSON.stringify(data, null, 2));
          } else if (sdp.type === "answer") {
            console.log(
              `[${performance.now().toFixed(2)}ms] Setting answer data.`
            );
            setAnswerData(JSON.stringify(data, null, 2));
          }
        }
      }
    };

    pc.current = newPc;
  };

  /**
   * Handles creating a call offer.
   */
  const handleCreateCall = async () => {
    console.log(
      `[${performance.now().toFixed(2)}ms] handleCreateCall started.`
    );
    if (!localStream) return alert("Please start your webcam first!");
    setCallMode("creating");
    initializePeerConnection();
    if (pc.current) {
      console.log(`[${performance.now().toFixed(2)}ms] Creating offer...`);
      const offer = await pc.current.createOffer();
      console.log(
        `[${performance
          .now()
          .toFixed(2)}ms] Offer created. Setting local description...`
      );
      await pc.current.setLocalDescription(offer);
      console.log(`[${performance.now().toFixed(2)}ms] Local description set.`);
    }
  };

  /**
   * Handles joining a call and creating an answer.
   */
  const handleJoinCall = async () => {
    console.log(`[${performance.now().toFixed(2)}ms] handleJoinCall started.`);
    if (!localStream) return alert("Please start your webcam first!");
    if (!offerData.trim()) return alert("Please paste the offer data first.");

    try {
      const { sdp: offerSdp, candidates: offerCandidates } =
        JSON.parse(offerData);
      setCallMode("joining");
      initializePeerConnection();
      if (pc.current) {
        console.log(
          `[${performance
            .now()
            .toFixed(2)}ms] Setting remote description from offer...`
        );
        await pc.current.setRemoteDescription(
          new RTCSessionDescription(offerSdp)
        );
        console.log(
          `[${performance
            .now()
            .toFixed(2)}ms] Remote description set. Creating answer...`
        );
        const answer = await pc.current.createAnswer();
        console.log(
          `[${performance
            .now()
            .toFixed(2)}ms] Answer created. Setting local description...`
        );
        await pc.current.setLocalDescription(answer);
        console.log(
          `[${performance.now().toFixed(2)}ms] Local description set. Adding ${
            offerCandidates.length
          } remote ICE candidates...`
        );

        for (const candidate of offerCandidates) {
          await pc.current.addIceCandidate(new RTCIceCandidate(candidate));
        }
        console.log(
          `[${performance
            .now()
            .toFixed(2)}ms] Finished adding remote ICE candidates.`
        );
      }
    } catch (e) {
      console.error(
        `[${performance.now().toFixed(2)}ms] Error in handleJoinCall:`,
        e
      );
      alert("Invalid Offer Data format.");
      setCallMode("idle");
    }
  };

  /**
   * Connects the peers using the answer data.
   */
  const handleConnect = async () => {
    console.log(`[${performance.now().toFixed(2)}ms] handleConnect started.`);
    if (!answerData.trim())
      return alert("Please paste the peer's answer data.");
    if (!pc.current) return alert("Peer connection not initialized.");

    try {
      const { sdp: answerSdp, candidates: answerCandidates } =
        JSON.parse(answerData);
      if (!pc.current.currentRemoteDescription) {
        console.log(
          `[${performance
            .now()
            .toFixed(2)}ms] Setting remote description from answer...`
        );
        await pc.current.setRemoteDescription(
          new RTCSessionDescription(answerSdp)
        );
        console.log(
          `[${performance.now().toFixed(2)}ms] Remote description set.`
        );
      }
      console.log(
        `[${performance.now().toFixed(2)}ms] Adding ${
          answerCandidates.length
        } remote ICE candidates from answer...`
      );
      for (const candidate of answerCandidates) {
        await pc.current.addIceCandidate(new RTCIceCandidate(candidate));
      }
      console.log(
        `[${performance
          .now()
          .toFixed(2)}ms] Finished adding remote ICE candidates from answer.`
      );
    } catch (e) {
      console.error(
        `[${performance.now().toFixed(2)}ms] Error in handleConnect:`,
        e
      );
      alert("Invalid Answer Data format.");
    }
  };

  /**
   * Resets the application state to hang up the call.
   */
  const hangUp = () => {
    console.log(`[${performance.now().toFixed(2)}ms] Hanging up call.`);
    pc.current?.close();
    pc.current = null;
    localStream?.getTracks().forEach((track) => track.stop());
    remoteStream?.getTracks().forEach((track) => track.stop());

    setLocalStream(null);
    setRemoteStream(null);
    setCallMode("idle");
    setOfferData("");
    setAnswerData("");
    setShowEndCallConfirm(false);
    setIsMuted(false);
    setIsVideoOff(false);
    setIsConnectionEstablished(false);
  };

  const toggleMic = () => {
    if (localStream) {
      localStream.getAudioTracks().forEach((track) => {
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

  // Effect to attach the remote stream to the video element
  useEffect(() => {
    if (remoteVideoRef.current && remoteStream) {
      remoteVideoRef.current.srcObject = remoteStream;
    }
  }, [remoteStream]);

  const copyToClipboard = (text: string) => {
    if (!text) return;
    navigator.clipboard
      .writeText(text)
      .then(() => alert("Copied to clipboard!"));
  };

  return (
    <div className="bg-gray-900 text-white min-h-screen flex flex-col items-center p-4 font-sans">
      <div className="w-full max-w-5xl mx-auto">
        <header className="text-center mb-6">
          <h1 className="text-4xl font-bold text-teal-400">
            Serverless WebRTC
          </h1>
          <p className="text-gray-400">A demo using manual signaling</p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="relative bg-black rounded-lg overflow-hidden shadow-lg aspect-video">
            <video
              ref={localVideoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            ></video>
            <span className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white text-sm px-2 py-1 rounded">
              You
            </span>
          </div>
          <div className="relative bg-black rounded-lg overflow-hidden shadow-lg aspect-video">
            <video
              ref={remoteVideoRef}
              autoPlay
              playsInline
              className="w-full h-full object-cover"
            ></video>
            <span className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white text-sm px-2 py-1 rounded">
              Friend
            </span>
            {!remoteStream && (
              <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75">
                <p className="text-gray-400">No connection yet</p>
              </div>
            )}
          </div>
        </div>

        <main className="bg-gray-800 p-6 rounded-lg shadow-2xl min-h-[280px]">
          {!isConnectionEstablished ? (
            <>
              {/* Initial State: Start webcam and choose mode */}
              {callMode === "idle" && (
                <div className="text-center">
                  <button
                    onClick={startWebcam}
                    disabled={!!localStream}
                    className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg font-semibold transition-colors duration-300 mb-4"
                  >
                    1. Start Webcam
                  </button>
                  <div className="flex justify-center gap-4">
                    <button
                      onClick={handleCreateCall}
                      disabled={!localStream}
                      className="px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-lg font-semibold"
                    >
                      Create Call
                    </button>
                    <button
                      onClick={() => setCallMode("joining")}
                      disabled={!localStream}
                      className="px-6 py-3 bg-teal-600 hover:bg-teal-700 disabled:bg-gray-600 rounded-lg font-semibold"
                    >
                      Join Call
                    </button>
                  </div>
                </div>
              )}

              {/* Creating a Call Workflow */}
              {callMode === "creating" && (
                <div className="flex flex-col gap-4">
                  <Step number={2} title="Copy & Send Offer">
                    <p className="text-sm text-gray-400 mb-2">
                      Send this entire block of text to the other person.
                    </p>
                    <textarea
                      value={offerData}
                      readOnly
                      className="w-full h-32 bg-gray-900 p-2 rounded font-mono text-xs"
                    ></textarea>
                    <button
                      onClick={() => copyToClipboard(offerData)}
                      className="mt-2 w-full p-2 bg-gray-600 hover:bg-gray-500 rounded"
                    >
                      Copy Offer
                    </button>
                  </Step>
                  <Step number={3} title="Paste Peer's Answer">
                    <p className="text-sm text-gray-400 mb-2">
                      Once they send their answer back, paste it here.
                    </p>
                    <textarea
                      value={answerData}
                      onChange={(e) => setAnswerData(e.target.value)}
                      className="w-full h-32 bg-gray-900 p-2 rounded font-mono text-xs"
                    ></textarea>
                  </Step>
                  <button
                    onClick={handleConnect}
                    className="w-full p-3 bg-green-500 hover:bg-green-600 rounded font-bold text-lg"
                  >
                    4. Connect
                  </button>
                </div>
              )}

              {/* Joining a Call Workflow */}
              {callMode === "joining" && (
                <div className="flex flex-col gap-4">
                  <Step number={2} title="Paste Peer's Offer">
                    <textarea
                      value={offerData}
                      onChange={(e) => setOfferData(e.target.value)}
                      className="w-full h-32 bg-gray-900 p-2 rounded font-mono text-xs"
                    ></textarea>
                    <button
                      onClick={handleJoinCall}
                      className="mt-2 w-full p-2 bg-teal-600 hover:bg-teal-700 rounded"
                    >
                      Create Answer from Offer
                    </button>
                  </Step>
                  {answerData && (
                    <Step number={3} title="Copy & Send Your Answer">
                      <p className="text-sm text-gray-400 mb-2">
                        Send this back to the person who created the call.
                      </p>
                      <textarea
                        value={answerData}
                        readOnly
                        className="w-full h-32 bg-gray-900 p-2 rounded font-mono text-xs"
                      ></textarea>
                      <button
                        onClick={() => copyToClipboard(answerData)}
                        className="mt-2 w-full p-2 bg-gray-600 hover:bg-gray-500 rounded"
                      >
                        Copy Your Answer
                      </button>
                    </Step>
                  )}
                </div>
              )}

              {/* Hangup Button visible during call setup */}
              {callMode !== "idle" && (
                <div className="text-center mt-6">
                  <button
                    onClick={hangUp}
                    className="px-8 py-3 bg-red-600 hover:bg-red-700 rounded-lg font-semibold"
                  >
                    Cancel
                  </button>
                </div>
              )}
            </>
          ) : (
            <div className="flex justify-center items-center gap-4 h-full">
              {/* In-call controls */}
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
                disabled
                className="p-4 rounded-full bg-gray-700 cursor-not-allowed opacity-50"
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
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 3a1 1 0 011 1v1.586l4.707 4.707a1 1 0 01-1.414 1.414L10 6.414 5.707 10.707a1 1 0 11-1.414-1.414L9 4.586V4a1 1 0 011-1z"
                    clipRule="evenodd"
                  ></path>
                  <path d="M3.5 6.5a1 1 0 011-1h11a1 1 0 010 2h-11a1 1 0 01-1-1z"></path>
                </svg>
              </button>
            </div>
          )}
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
