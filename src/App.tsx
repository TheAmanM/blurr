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

  // useRef for mutable objects that don't trigger re-renders
  const pc = useRef<RTCPeerConnection | null>(null);
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);

  // Textarea state for manual signaling
  const [offerData, setOfferData] = useState("");
  const [answerData, setAnswerData] = useState("");
  const [remoteCandidates, setRemoteCandidates] = useState("");

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
          // BUG FIX: Differentiate between setting offer and answer data based on SDP type
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

        // Add candidates from the offer
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
    setRemoteCandidates("");
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

        <main className="bg-gray-800 p-6 rounded-lg shadow-2xl">
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

          {/* Hangup Button visible during call */}
          {callMode !== "idle" && (
            <div className="text-center mt-6">
              <button
                onClick={hangUp}
                className="px-8 py-3 bg-red-600 hover:bg-red-700 rounded-lg font-semibold"
              >
                Hang Up
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
