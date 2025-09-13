import React, { useState, useRef, useEffect } from "react";

// --- STUN servers for NAT traversal ---
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
  const [callMode, setCallMode] = useState<"idle" | "creating" | "joining">(
    "idle"
  );
  const [isMuted, setIsMuted] = useState(false);
  const [isVideoOff, setIsVideoOff] = useState(false);
  const [showEndCallConfirm, setShowEndCallConfirm] = useState(false);
  const [isConnectionEstablished, setIsConnectionEstablished] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);

  // Refs for non-re-rendering objects
  const pc = useRef<RTCPeerConnection | null>(null);
  const localCamStreamRef = useRef<MediaStream | null>(null);
  const screenSenderRef = useRef<RTCRtpSender | null>(null);

  // Refs for video elements
  const localCamVideoRef = useRef<HTMLVideoElement>(null);
  const localScreenVideoRef = useRef<HTMLVideoElement>(null);
  const remoteCamVideoRef = useRef<HTMLVideoElement>(null);
  const remoteScreenVideoRef = useRef<HTMLVideoElement>(null);

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
      localCamStreamRef.current = stream;
      setLocalCamStream(stream);
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

    // Add local camera tracks to the connection initially
    if (localCamStreamRef.current) {
      console.log(
        `[${performance.now().toFixed(2)}ms] Adding local camera tracks.`
      );
      localCamStreamRef.current.getTracks().forEach((track) => {
        newPc.addTrack(track, localCamStreamRef.current!);
      });
    }

    // Handles incoming tracks from the remote peer
    newPc.ontrack = (event) => {
      const track = event.track;
      console.log(
        `[${performance.now().toFixed(2)}ms] Remote track received: ${
          track.kind
        }`
      );

      const stream = event.streams[0] || new MediaStream([track]);

      if (track.kind === "video") {
        const settings = track.getSettings();
        if (settings.displaySurface) {
          setRemoteScreenStream(stream);
        } else {
          setRemoteCamStream(stream);
        }
      } else if (track.kind === "audio") {
        setRemoteCamStream(stream);
      }
    };

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
        setIsUpdating(false); // Finish update process
        // Clear signaling data once fully connected
        setOfferData("");
        setAnswerData("");
      }
    };

    // Handles re-negotiation when tracks are added/removed
    newPc.onnegotiationneeded = async () => {
      console.log(`[${performance.now().toFixed(2)}ms] Negotiation needed.`);
      if (newPc.signalingState !== "stable") {
        console.log("--> Signaling state is not stable, skipping negotiation.");
        return;
      }
      try {
        setIsUpdating(true); // Show the signaling UI for the user initiating
        const offer = await newPc.createOffer();
        await newPc.setLocalDescription(offer);
      } catch (err) {
        console.error("Error creating negotiation offer:", err);
      }
    };

    let candidates: RTCIceCandidateInit[] = [];
    newPc.onicecandidate = (event) => {
      if (event.candidate) {
        candidates.push(event.candidate.toJSON());
      }
    };

    newPc.onicegatheringstatechange = () => {
      if (newPc.iceGatheringState === "complete") {
        const sdp = newPc.localDescription;
        if (sdp) {
          const data = { sdp, candidates: [...candidates] };
          if (sdp.type === "offer") {
            setOfferData(JSON.stringify(data, null, 2));
          } else if (sdp.type === "answer") {
            setAnswerData(JSON.stringify(data, null, 2));
          }
        }
      }
    };

    pc.current = newPc;
  };

  const handleCreateCall = async () => {
    if (!localCamStream) return alert("Please start your webcam first!");
    setCallMode("creating");
    initializePeerConnection();
    if (pc.current) {
      const offer = await pc.current.createOffer();
      await pc.current.setLocalDescription(offer);
    }
  };

  const handleJoinCall = async () => {
    if (!localCamStream) return alert("Please start your webcam first!");
    if (!offerData.trim()) return alert("Please paste the offer data first.");

    try {
      const { sdp: offerSdp, candidates: offerCandidates } =
        JSON.parse(offerData);

      if (!pc.current) {
        setCallMode("joining");
        initializePeerConnection();
      }

      if (pc.current) {
        await pc.current.setRemoteDescription(
          new RTCSessionDescription(offerSdp)
        );
        const answer = await pc.current.createAnswer();
        await pc.current.setLocalDescription(answer);

        for (const candidate of offerCandidates) {
          await pc.current.addIceCandidate(new RTCIceCandidate(candidate));
        }
      }
    } catch (e) {
      alert("Invalid Offer Data format.");
      if (!pc.current) setCallMode("idle");
    }
  };

  const handleConnect = async () => {
    if (!answerData.trim())
      return alert("Please paste the peer's answer data.");
    if (!pc.current) return alert("Peer connection not initialized.");

    try {
      const { sdp: answerSdp, candidates: answerCandidates } =
        JSON.parse(answerData);
      await pc.current.setRemoteDescription(
        new RTCSessionDescription(answerSdp)
      );
      for (const candidate of answerCandidates) {
        await pc.current.addIceCandidate(new RTCIceCandidate(candidate));
      }
    } catch (e) {
      alert("Invalid Answer Data format.");
    }
  };

  const hangUp = () => {
    pc.current?.close();
    localCamStreamRef.current?.getTracks().forEach((track) => track.stop());
    localScreenStream?.getTracks().forEach((track) => track.stop());
    setLocalCamStream(null);
    setLocalScreenStream(null);
    setRemoteCamStream(null);
    setRemoteScreenStream(null);
    setCallMode("idle");
    setOfferData("");
    setAnswerData("");
    setShowEndCallConfirm(false);
    setIsMuted(false);
    setIsVideoOff(false);
    setIsConnectionEstablished(false);
    setIsUpdating(false);
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
    if (!pc.current) return;

    if (screenSenderRef.current) {
      console.log(`[${performance.now().toFixed(2)}ms] Stopping screen share.`);
      pc.current.removeTrack(screenSenderRef.current);
      localScreenStream?.getTracks().forEach((track) => track.stop());
      setLocalScreenStream(null);
      screenSenderRef.current = null;
    } else {
      console.log(`[${performance.now().toFixed(2)}ms] Starting screen share.`);
      try {
        const screenStream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
        });
        const screenTrack = screenStream.getVideoTracks()[0];
        screenSenderRef.current = pc.current.addTrack(
          screenTrack,
          screenStream
        );
        setLocalScreenStream(screenStream);
        screenTrack.onended = () => {
          if (screenSenderRef.current) {
            toggleScreenShare();
          }
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
          <p className="text-gray-400">No connection yet</p>
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
        !isConnectionEstablished
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

  const renderSignalingWizard = () => (
    <>
      {isUpdating && (
        <div className="text-center p-2 rounded-lg bg-yellow-900 border border-yellow-400 mb-4">
          <h3 className="text-yellow-300 font-bold">
            Connection Update Required
          </h3>
          <p className="text-xs text-yellow-200">
            A user has started/stopped screen sharing. Exchange the data below
            to see the change.
          </p>
        </div>
      )}
      {callMode === "idle" && (
        <div className="text-center">
          <button
            onClick={startWebcam}
            disabled={!!localCamStream}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg font-semibold transition-colors duration-300 mb-4"
          >
            1. Start Webcam
          </button>
          <div className="flex justify-center gap-4">
            <button
              onClick={handleCreateCall}
              disabled={!localCamStream}
              className="px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-lg font-semibold"
            >
              Create Call
            </button>
            <button
              onClick={() => setCallMode("joining")}
              disabled={!localCamStream}
              className="px-6 py-3 bg-teal-600 hover:bg-teal-700 disabled:bg-gray-600 rounded-lg font-semibold"
            >
              Join Call
            </button>
          </div>
        </div>
      )}
      {callMode === "creating" && (
        <div className="flex flex-col gap-4">
          <Step number={2} title="Copy & Send Offer">
            <textarea
              value={offerData}
              readOnly
              className="w-full h-24 bg-gray-900 p-2 rounded font-mono text-xs"
            />
            <button
              onClick={() => copyToClipboard(offerData)}
              className="mt-2 w-full p-2 bg-gray-600 hover:bg-gray-500 rounded"
            >
              Copy Offer
            </button>
          </Step>
          <Step number={3} title="Paste Peer's Answer">
            <textarea
              value={answerData}
              onChange={(e) => setAnswerData(e.target.value)}
              className="w-full h-24 bg-gray-900 p-2 rounded font-mono text-xs"
            />
          </Step>
          <button
            onClick={handleConnect}
            className="w-full p-3 bg-green-500 hover:bg-green-600 rounded font-bold text-lg"
          >
            4. Connect
          </button>
        </div>
      )}
      {callMode === "joining" && (
        <div className="flex flex-col gap-4">
          <Step number={2} title="Paste Peer's Offer">
            <textarea
              value={offerData}
              onChange={(e) => setOfferData(e.target.value)}
              className="w-full h-24 bg-gray-900 p-2 rounded font-mono text-xs"
            />
            <button
              onClick={handleJoinCall}
              className="mt-2 w-full p-2 bg-teal-600 hover:bg-teal-700 rounded"
            >
              Create Answer from Offer
            </button>
          </Step>
          {answerData && (
            <Step number={3} title="Copy & Send Your Answer">
              <textarea
                value={answerData}
                readOnly
                className="w-full h-24 bg-gray-900 p-2 rounded font-mono text-xs"
              />
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
  );

  const renderMediaControls = () => (
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
        className={`p-4 rounded-full transition-colors ${
          !!localScreenStream ? "bg-blue-500" : "bg-gray-600 hover:bg-gray-500"
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
        onClick={() => setIsUpdating(true)}
        className="p-4 rounded-full bg-yellow-600 hover:bg-yellow-500"
        title="Process connection update"
      >
        <svg
          className="w-6 h-6"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M4 4v5h5M20 20v-5h-5M4 4l5 5M20 20l-5-5"
          ></path>
        </svg>
      </button>
      <button
        onClick={() => setShowEndCallConfirm(true)}
        className="p-4 rounded-full bg-red-600 hover:bg-red-700"
      >
        <svg className="w-6 h-6" fill="currentColor" viewBox="0 -256 1792 1792">
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
          <p className="text-gray-400">A demo using manual signaling</p>
        </header>

        <div className="flex-grow min-h-0">{renderVideos()}</div>

        <main className="flex-shrink-0 bg-gray-800 p-6 rounded-lg shadow-2xl mt-4">
          {!isConnectionEstablished || isUpdating
            ? renderSignalingWizard()
            : renderMediaControls()}
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
