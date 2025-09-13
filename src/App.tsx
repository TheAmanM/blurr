import { useState, useRef, useEffect } from "react";

// --- STUN servers for NAT traversal (still needed) ---
const servers = {
  iceServers: [
    {
      urls: ["stun:stun1.l.google.com:19302", "stun:stun2.l.google.com:19302"],
    },
  ],
  iceCandidatePoolSize: 10,
};

function App() {
  // State variables
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [isCallStarted, setIsCallStarted] = useState(false);

  // useRef for mutable objects that don't trigger re-renders
  const pc = useRef<RTCPeerConnection | null>(null);
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);

  // Textarea state for manual signaling
  const [offerSDP, setOfferSDP] = useState("");
  const [answerSDP, setAnswerSDP] = useState("");
  const [localCandidates, setLocalCandidates] = useState("");
  const [remoteCandidates, setRemoteCandidates] = useState("");

  /**
   * Initializes the webcam and microphone stream.
   */
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      setLocalStream(stream);
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error("Error accessing media devices.", error);
      alert("Could not access webcam. Please check permissions.");
    }
  };

  /**
   * Initializes the RTCPeerConnection object for both peers.
   */
  const initializePeerConnection = () => {
    pc.current = new RTCPeerConnection(servers);

    // Add local stream tracks to the connection
    if (localStream) {
      localStream.getTracks().forEach((track) => {
        pc.current?.addTrack(track, localStream);
      });
    }

    // Handle incoming remote stream
    pc.current.ontrack = (event) => {
      const newStream = new MediaStream();
      event.streams[0].getTracks().forEach((track) => {
        newStream.addTrack(track);
      });
      setRemoteStream(newStream);
    };

    // Collect ICE candidates
    pc.current.onicecandidate = (event) => {
      if (event.candidate) {
        setLocalCandidates(
          (prev) => prev + JSON.stringify(event.candidate) + "\n"
        );
      }
    };
  };

  /**
   * Peer 1: Creates an SDP offer.
   */
  const createOffer = async () => {
    if (!localStream) {
      return alert("Please start your webcam first!");
    }
    initializePeerConnection();
    if (pc.current) {
      const offerDescription = await pc.current.createOffer();
      await pc.current.setLocalDescription(offerDescription);
      setOfferSDP(JSON.stringify(offerDescription));
      setIsCallStarted(true);
    }
  };

  /**
   * Peer 2: Creates an SDP answer from the offer.
   */
  const createAnswer = async () => {
    if (!localStream) {
      return alert("Please start your webcam first!");
    }
    if (!offerSDP) {
      return alert("Please paste the offer from the other peer first.");
    }
    initializePeerConnection();
    if (pc.current) {
      try {
        const offer = JSON.parse(offerSDP);
        await pc.current.setRemoteDescription(new RTCSessionDescription(offer));
        const answerDescription = await pc.current.createAnswer();
        await pc.current.setLocalDescription(answerDescription);
        setAnswerSDP(JSON.stringify(answerDescription));
        setIsCallStarted(true);
      } catch (error) {
        console.error(error);
        alert("Invalid offer SDP. Please check the pasted value.");
      }
    }
  };

  /**
   * Peer 1: Sets the remote description using the answer.
   */
  const setRemoteAnswer = async () => {
    if (!answerSDP) {
      return alert("Please paste the answer from the other peer first.");
    }
    if (pc.current) {
      try {
        const answer = JSON.parse(answerSDP);
        if (!pc.current.currentRemoteDescription) {
          await pc.current.setRemoteDescription(
            new RTCSessionDescription(answer)
          );
        }
      } catch (error) {
        console.error(error);
        alert("Invalid answer SDP. Please check the pasted value.");
      }
    }
  };

  /**
   * Both Peers: Add ICE candidates received from the other peer.
   */
  const addRemoteCandidates = async () => {
    if (!remoteCandidates.trim() || !pc.current) {
      return alert("Paste candidates first or start a call.");
    }
    try {
      const candidates = remoteCandidates.trim().split("\n");
      for (const candidateStr of candidates) {
        if (candidateStr) {
          await pc.current.addIceCandidate(
            new RTCIceCandidate(JSON.parse(candidateStr))
          );
        }
      }
      setRemoteCandidates(""); // Clear after adding
    } catch (error) {
      console.error(error);
      alert("Invalid ICE candidate format.");
    }
  };

  // Effect to attach the remote stream to the video element
  useEffect(() => {
    if (remoteVideoRef.current && remoteStream) {
      remoteVideoRef.current.srcObject = remoteStream;
    }
  }, [remoteStream]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard
      .writeText(text)
      .then(() => alert("Copied to clipboard!"));
  };

  return (
    <div className="bg-gray-900 text-white min-h-screen flex flex-col items-center p-4 font-sans">
      <div className="w-full max-w-6xl mx-auto">
        <header className="text-center mb-6">
          <h1 className="text-4xl font-bold text-teal-400">
            Serverless WebRTC
          </h1>
          <p className="text-gray-400">A demo using manual signaling</p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* Local Video */}
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
          {/* Remote Video */}
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
          <div className="text-center mb-6">
            <button
              onClick={startWebcam}
              disabled={!!localStream}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold transition-colors duration-300"
            >
              1. Start Webcam
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Peer 1 (Caller) Column */}
            <div className="flex flex-col gap-4">
              <h2 className="text-xl font-bold text-center text-teal-300">
                Peer 1 (Caller)
              </h2>
              <button
                onClick={createOffer}
                disabled={!localStream || isCallStarted}
                className="w-full p-3 bg-green-600 hover:bg-green-700 rounded disabled:bg-gray-600"
              >
                2. Create Offer
              </button>
              <textarea
                value={offerSDP}
                readOnly
                placeholder="Offer SDP will appear here..."
                className="w-full h-24 bg-gray-700 p-2 rounded focus:outline-none focus:ring-2 focus:ring-teal-500 font-mono text-xs"
              ></textarea>
              <button
                onClick={() => copyToClipboard(offerSDP)}
                disabled={!offerSDP}
                className="w-full p-2 bg-gray-600 hover:bg-gray-500 rounded disabled:opacity-50"
              >
                Copy Offer
              </button>

              <textarea
                onChange={(e) => setAnswerSDP(e.target.value)}
                placeholder="5. Paste Answer SDP here..."
                className="w-full h-24 bg-gray-700 p-2 rounded focus:outline-none focus:ring-2 focus:ring-teal-500 font-mono text-xs"
              ></textarea>
              <button
                onClick={setRemoteAnswer}
                disabled={!answerSDP}
                className="w-full p-3 bg-teal-600 hover:bg-teal-700 rounded disabled:bg-gray-600"
              >
                6. Set Remote Answer
              </button>
            </div>

            {/* Peer 2 (Receiver) Column */}
            <div className="flex flex-col gap-4">
              <h2 className="text-xl font-bold text-center text-teal-300">
                Peer 2 (Receiver)
              </h2>
              <textarea
                onChange={(e) => setOfferSDP(e.target.value)}
                placeholder="3. Paste Offer SDP here..."
                className="w-full h-24 bg-gray-700 p-2 rounded focus:outline-none focus:ring-2 focus:ring-teal-500 font-mono text-xs"
              ></textarea>
              <button
                onClick={createAnswer}
                disabled={!localStream || isCallStarted}
                className="w-full p-3 bg-green-600 hover:bg-green-700 rounded disabled:bg-gray-600"
              >
                4. Create Answer
              </button>
              <textarea
                value={answerSDP}
                readOnly
                placeholder="Answer SDP will appear here..."
                className="w-full h-24 bg-gray-700 p-2 rounded focus:outline-none focus:ring-2 focus:ring-teal-500 font-mono text-xs"
              ></textarea>
              <button
                onClick={() => copyToClipboard(answerSDP)}
                disabled={!answerSDP}
                className="w-full p-2 bg-gray-600 hover:bg-gray-500 rounded disabled:opacity-50"
              >
                Copy Answer
              </button>
            </div>
          </div>

          <hr className="my-8 border-gray-600" />

          {/* ICE Candidate Exchange Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="flex flex-col gap-4">
              <h3 className="text-lg font-semibold text-center">
                Your ICE Candidates
              </h3>
              <textarea
                value={localCandidates}
                readOnly
                placeholder="Your candidates for the other peer..."
                className="w-full h-32 bg-gray-700 p-2 rounded font-mono text-xs"
              ></textarea>
              <button
                onClick={() => copyToClipboard(localCandidates)}
                disabled={!localCandidates}
                className="w-full p-2 bg-gray-600 hover:bg-gray-500 rounded disabled:opacity-50"
              >
                Copy My Candidates
              </button>
            </div>
            <div className="flex flex-col gap-4">
              <h3 className="text-lg font-semibold text-center">
                Friend's ICE Candidates
              </h3>
              <textarea
                value={remoteCandidates}
                onChange={(e) => setRemoteCandidates(e.target.value)}
                placeholder="Paste candidates from other peer here (one per line)..."
                className="w-full h-32 bg-gray-700 p-2 rounded font-mono text-xs"
              ></textarea>
              <button
                onClick={addRemoteCandidates}
                disabled={!remoteCandidates}
                className="w-full p-3 bg-teal-600 hover:bg-teal-700 rounded disabled:bg-gray-600"
              >
                Add Friend's Candidates
              </button>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
