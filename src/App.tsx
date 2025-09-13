// Firebase imports for signaling
import { initializeApp } from "firebase/app";
import {
  getFirestore,
  doc,
  getDoc,
  setDoc,
  updateDoc,
  onSnapshot,
  collection,
  addDoc,
  getDocs,
  deleteDoc,
  query,
} from "firebase/firestore";
import {
  getAuth,
  signInAnonymously,
  onAuthStateChanged,
  signInWithCustomToken,
} from "firebase/auth";
import type { User } from "firebase/auth";
import { useEffect, useRef, useState } from "react";

// --- Firebase Configuration ---
// These global variables are provided by the environment.
const appId = "default-app-id";
const firebaseConfig = {
  apiKey: "DEMO",
  authDomain: "DEMO",
  projectId: "DEMO",
};
const initialAuthToken = null;

// --- Initialize Firebase ---
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const auth = getAuth(app);

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
  // State variables
  const [user, setUser] = useState<User | null>(null);
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [callId, setCallId] = useState("");
  const [callInput, setCallInput] = useState("");
  const [callInProgress, setCallInProgress] = useState(false);
  const [showCopied, setShowCopied] = useState(false);

  // useRef for mutable objects that don't trigger re-renders
  const pc = useRef<RTCPeerConnection | null>(undefined);
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);

  // --- Firebase Authentication ---
  useEffect(() => {
    const initAuth = async () => {
      try {
        if (initialAuthToken) {
          await signInWithCustomToken(auth, initialAuthToken);
        } else {
          await signInAnonymously(auth);
        }
      } catch (error) {
        console.error("Firebase Auth Error:", error);
      }
    };

    initAuth();

    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      if (currentUser) {
        setUser(currentUser);
      } else {
        setUser(null);
      }
    });

    return () => unsubscribe();
  }, []);

  // --- Core WebRTC Logic ---

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
    }
  };

  /**
   * Creates a new call, generates an offer, and sets up listeners for the answer and ICE candidates.
   */
  const createCall = async () => {
    if (!localStream) {
      alert("Please start your webcam first!");
      return;
    }

    setCallInProgress(true);

    // Reference to the 'calls' collection in Firestore
    const callCollection = collection(db, "calls");
    const callDoc = await addDoc(callCollection, {});
    const callDocId = callDoc.id;
    setCallId(callDocId);

    // Create a new RTCPeerConnection
    pc.current = new RTCPeerConnection(servers);

    // Add local stream tracks to the connection
    localStream.getTracks().forEach((track) => {
      if (pc.current) {
        pc.current.addTrack(track, localStream);
      }
    });

    // Set up listeners for remote stream and ICE candidates
    pc.current.ontrack = (event) => {
      event.streams[0].getTracks().forEach((track) => {
        if (remoteStream) {
          remoteStream.addTrack(track);
        } else {
          const newStream = new MediaStream();
          newStream.addTrack(track);
          setRemoteStream(newStream);
        }
      });
    };

    const offerCandidatesCollection = collection(
      db,
      "calls",
      callDocId,
      "offerCandidates"
    );
    const answerCandidatesCollection = collection(
      db,
      "calls",
      callDocId,
      "answerCandidates"
    );

    pc.current.onicecandidate = (event) => {
      event.candidate &&
        addDoc(offerCandidatesCollection, event.candidate.toJSON());
    };

    // Create offer
    const offerDescription = await pc.current.createOffer();
    await pc.current.setLocalDescription(offerDescription);

    const offer = {
      sdp: offerDescription.sdp,
      type: offerDescription.type,
    };

    await setDoc(callDoc, { offer });

    // Listen for the answer from the other peer
    onSnapshot(callDoc, (snapshot) => {
      const data = snapshot.data();
      if (pc.current && !pc.current.currentRemoteDescription && data?.answer) {
        const answerDescription = new RTCSessionDescription(data.answer);
        pc.current.setRemoteDescription(answerDescription);
      }
    });

    // Listen for ICE candidates from the other peer
    onSnapshot(answerCandidatesCollection, (snapshot) => {
      snapshot.docChanges().forEach((change) => {
        if (change.type === "added") {
          const candidate = new RTCIceCandidate(change.doc.data());
          if (pc.current) {
            pc.current.addIceCandidate(candidate);
          }
        }
      });
    });
  };

  /**
   * Joins an existing call using the provided call ID.
   */
  const joinCall = async () => {
    if (!localStream) {
      alert("Please start your webcam first!");
      return;
    }
    if (!callInput.trim()) {
      alert("Please enter a call ID to join!");
      return;
    }

    setCallInProgress(true);
    setCallId(callInput);

    const callDocRef = doc(db, "calls", callInput);
    const callDocSnapshot = await getDoc(callDocRef);
    const callData = callDocSnapshot.data();

    if (!callData || !callData.offer) {
      alert("Invalid Call ID. Please check and try again.");
      setCallInProgress(false);
      return;
    }

    const offerCandidatesCollection = collection(
      db,
      "calls",
      callInput,
      "offerCandidates"
    );
    const answerCandidatesCollection = collection(
      db,
      "calls",
      callInput,
      "answerCandidates"
    );

    pc.current = new RTCPeerConnection(servers);

    localStream.getTracks().forEach((track) => {
      if (pc.current) {
        pc.current.addTrack(track, localStream);
      }
    });

    pc.current.onicecandidate = (event) => {
      event.candidate &&
        addDoc(answerCandidatesCollection, event.candidate.toJSON());
    };

    pc.current.ontrack = (event) => {
      event.streams[0].getTracks().forEach((track) => {
        if (remoteStream) {
          remoteStream.addTrack(track);
        } else {
          const newStream = new MediaStream();
          newStream.addTrack(track);
          setRemoteStream(newStream);
        }
      });
    };

    const offerDescription = new RTCSessionDescription(callData.offer);
    await pc.current.setRemoteDescription(offerDescription);

    const answerDescription = await pc.current.createAnswer();
    await pc.current.setLocalDescription(answerDescription);

    const answer = {
      type: answerDescription.type,
      sdp: answerDescription.sdp,
    };

    await updateDoc(callDocRef, { answer });

    onSnapshot(offerCandidatesCollection, (snapshot) => {
      snapshot.docChanges().forEach((change) => {
        if (change.type === "added") {
          const candidate = new RTCIceCandidate(change.doc.data());
          if (pc.current) {
            pc.current.addIceCandidate(candidate);
          }
        }
      });
    });
  };

  /**
   * Ends the call, closes the connection, and resets state.
   */
  const hangUp = async () => {
    if (pc.current) {
      pc.current.close();
      pc.current = null;
    }

    if (localStream) {
      localStream.getTracks().forEach((track) => track.stop());
      setLocalStream(null);
    }

    if (remoteStream) {
      remoteStream.getTracks().forEach((track) => track.stop());
      setRemoteStream(null);
    }

    // Clean up Firestore document if this user created the call
    if (callId && !callInput) {
      try {
        const callDocRef = doc(db, "calls", callId);
        const offerCandidatesQuery = query(
          collection(callDocRef, "offerCandidates")
        );
        const answerCandidatesQuery = query(
          collection(callDocRef, "answerCandidates")
        );

        const [offerCandidatesSnapshot, answerCandidatesSnapshot] =
          await Promise.all([
            getDocs(offerCandidatesQuery),
            getDocs(answerCandidatesQuery),
          ]);

        const deletePromises: Promise<void>[] = [];
        offerCandidatesSnapshot.forEach((doc) =>
          deletePromises.push(deleteDoc(doc.ref))
        );
        answerCandidatesSnapshot.forEach((doc) =>
          deletePromises.push(deleteDoc(doc.ref))
        );

        await Promise.all(deletePromises);
        await deleteDoc(callDocRef);
      } catch (error) {
        console.error("Error cleaning up firestore document:", error);
      }
    }

    setCallId("");
    setCallInput("");
    setCallInProgress(false);
  };

  useEffect(() => {
    if (remoteVideoRef.current && remoteStream) {
      remoteVideoRef.current.srcObject = remoteStream;
    }
  }, [remoteStream]);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(callId).then(() => {
      setShowCopied(true);
      setTimeout(() => setShowCopied(false), 2000);
    });
  };

  // --- UI Rendering ---
  return (
    <div className="bg-gray-900 text-white min-h-screen flex flex-col items-center justify-center p-4 font-sans">
      <div className="w-full max-w-5xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-teal-400">ReactRTC</h1>
          <p className="text-gray-400">A Real-Time Video Chat App</p>
        </header>

        <main className="bg-gray-800 p-6 rounded-lg shadow-2xl">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
                  <p className="text-gray-400">Waiting for friend to join...</p>
                </div>
              )}
            </div>
          </div>

          <div className="mt-8">
            {/* Controls */}
            {!callInProgress ? (
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <button
                  onClick={startWebcam}
                  disabled={!!localStream}
                  className="w-full sm:w-auto px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold transition-colors duration-300 flex items-center justify-center gap-2"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path d="M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zM14.553 4.372A1 1 0 0116 5.143v9.714a1 1 0 01-1.447.894l-3.553-2.132V6.504l3.553-2.132z" />
                  </svg>
                  Start Webcam
                </button>
                <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto">
                  <button
                    onClick={createCall}
                    disabled={!localStream}
                    className="w-full sm:w-auto px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold transition-colors duration-300 flex items-center justify-center gap-2"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                      />
                    </svg>
                    Create Call
                  </button>
                  <div className="flex w-full sm:w-auto">
                    <input
                      value={callInput}
                      onChange={(e) => setCallInput(e.target.value)}
                      placeholder="Enter Call ID"
                      className="flex-grow bg-gray-700 text-white placeholder-gray-400 px-4 py-3 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-teal-500"
                    />
                    <button
                      onClick={joinCall}
                      disabled={!localStream || !callInput}
                      className="px-6 py-3 bg-teal-600 hover:bg-teal-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-r-lg font-semibold transition-colors duration-300"
                    >
                      Join
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center">
                <button
                  onClick={hangUp}
                  className="px-8 py-4 bg-red-600 hover:bg-red-700 rounded-lg font-semibold transition-colors duration-300 flex items-center gap-2"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M16 8l2-2m0 0l2 2m-2-2v2.5M20 12c0 4.418-3.582 8-8 8s-8-3.582-8-8 3.582-8 8-8 8 3.582 8 8zm-8 4l-2.293-2.293a1 1 0 010-1.414l.293-.293a1 1 0 011.414 0L12 11.586l2.293-2.293a1 1 0 011.414 0l.293.293a1 1 0 010 1.414L13.414 13l2.293 2.293a1 1 0 010 1.414l-.293.293a1 1 0 01-1.414 0L12 14.414l-2.293 2.293a1 1 0 01-1.414 0l-.293-.293a1 1 0 010-1.414L10.586 13z"
                    />
                  </svg>
                  Hang Up
                </button>
              </div>
            )}

            {callId && !callInput && (
              <div className="mt-6 p-4 bg-gray-700 rounded-lg text-center relative">
                <p className="text-sm text-gray-300">
                  Share this Call ID with your friend:
                </p>
                <div className="flex items-center justify-center mt-2">
                  <p className="text-lg font-mono text-teal-300 mr-4">
                    {callId}
                  </p>
                  <button
                    onClick={copyToClipboard}
                    className="p-2 bg-gray-600 hover:bg-gray-500 rounded-lg transition-colors duration-200"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
                      <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2H6zM8 7a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
                    </svg>
                  </button>
                </div>
                {showCopied && (
                  <div className="absolute -top-10 left-1/2 -translate-x-1/2 bg-green-500 text-white text-xs font-bold px-3 py-1 rounded-full shadow-lg">
                    Copied!
                  </div>
                )}
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
