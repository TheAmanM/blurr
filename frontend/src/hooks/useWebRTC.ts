import { useState, useRef, useCallback, useEffect } from "react";
import {
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
  DocumentReference,
  QuerySnapshot,
} from "firebase/firestore";
import { db } from "../lib/firebase";
import { toast } from "sonner";

const servers = {
  iceServers: [
    {
      urls: ["stun:stun1.l.google.com:19302", "stun:stun2.l.google.com:19302"],
    },
  ],
  iceCandidatePoolSize: 10,
};

type CallStatus =
  | "idle"
  | "creating"
  | "waiting"
  | "connecting"
  | "connected"
  | "error";

export const useWebRTC = () => {
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [isSharingScreen, setIsSharingScreen] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isVideoOff, setIsVideoOff] = useState(false);
  const [callId, setCallId] = useState<string | null>(null);
  const [callStatus, setCallStatus] = useState<CallStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const pc = useRef<RTCPeerConnection | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const videoSenderRef = useRef<RTCRtpSender | null>(null);
  const signalingUnsubscribers = useRef<any[]>([]);
  const callIdRef = useRef(callId);

  useEffect(() => {
    callIdRef.current = callId;
  }, [callId]);

  const setupSignalingListeners = useCallback(
    (callDocRef: DocumentReference, isOfferer: boolean) => {
      const mainUnsubscriber = onSnapshot(callDocRef, async (snapshot: any) => {
        const data = snapshot.data();
        if (!pc.current) return;

        if (data?.answer && pc.current.signalingState === "have-local-offer") {
          const answerDescription = new RTCSessionDescription(data.answer);
          await pc.current.setRemoteDescription(answerDescription);
        }

        if (data?.offer) {
          if (isOfferer && pc.current.connectionState !== "connected") return;
          if (pc.current.currentRemoteDescription?.sdp === data.offer.sdp)
            return;

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
    },
    []
  );

  const initializePeerConnection = useCallback(
    (currentCallId: string, isOfferer: boolean) => {
      const newPc = new RTCPeerConnection(servers);

      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach((track) => {
          const sender = newPc.addTrack(track, localStreamRef.current!);
          if (track.kind === "video") {
            videoSenderRef.current = sender;
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
        if (!isOfferer && newPc.connectionState !== "connected") return;
        if (newPc.signalingState !== "stable" || !currentCallId) return;

        try {
          const offer = await newPc.createOffer();
          await newPc.setLocalDescription(offer);
          if (newPc.localDescription) {
            const callDocRef = doc(db, "calls", currentCallId);
            await updateDoc(callDocRef, {
              offer: newPc.localDescription.toJSON(),
            });
          }
        } catch (err) {
          console.error("Error during negotiation:", err);
        }
      };

      pc.current = newPc;
    },
    []
  );

  const startWebcam = useCallback(async () => {
    try {
      // 1. Get the user's camera
      const rawStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });

      // 2. Create a new RTCPeerConnection to the backend
      const pcToBackend = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      });

      // 3. Add the camera track to the backend connection
      rawStream.getTracks().forEach(track => {
        pcToBackend.addTrack(track, rawStream);
      });

      // 4. Wait for the processed (blurred) stream from the backend
      const processedStreamPromise = new Promise<MediaStream>((resolve, reject) => {
        pcToBackend.ontrack = (event) => {
          if (event.streams && event.streams[0]) {
            resolve(event.streams[0]);
          }
        };
        pcToBackend.onconnectionstatechange = () => {
          if (["failed", "disconnected", "closed"].includes(pcToBackend.connectionState)) {
            reject(new Error("Connection to backend failed"));
          }
        };
      });

      // 5. Create and send offer to FastAPI backend
      const offer = await pcToBackend.createOffer();
      await pcToBackend.setLocalDescription(offer);
      const response = await fetch('http://localhost:8000/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sdp: offer.sdp, type: offer.type })
      });
      if (!response.ok) {
        throw new Error(`Backend responded with ${response.status}`);
      }
      const answer = await response.json();
      await pcToBackend.setRemoteDescription(new RTCSessionDescription(answer));

      // 6. Wait for the processed stream
      const processedStream = await processedStreamPromise;

      // 7. Set the processed stream as the local stream
      localStreamRef.current = processedStream;
      setLocalStream(processedStream);
      return processedStream;
    } catch (err) {
      console.error("Error accessing or processing camera:", err);
      toast.error("Could not access or process webcam. Please check permissions and backend.");
      return null;
    }
  }, []);

  const hangUp = useCallback(async () => {
    pc.current?.close();

    if (callIdRef.current) {
      const callDocRef = doc(db, "calls", callIdRef.current);
      const batch = writeBatch(db);
      batch.delete(callDocRef);
      await batch.commit();
    }

    signalingUnsubscribers.current.forEach((unsub) => unsub());
    signalingUnsubscribers.current = [];

    localStreamRef.current?.getTracks().forEach((track) => track.stop());

    setLocalStream(null);
    setRemoteStream(null);
    setCallId(null);
    setCallStatus("idle");
    setError(null);
    setIsMuted(false);
    setIsVideoOff(false);
    setIsSharingScreen(false);

    localStreamRef.current = null;
    videoSenderRef.current = null;
    pc.current = null;
  }, []);

  const createCall = useCallback(async () => {
    let stream = localStreamRef.current;
    if (!stream) {
      stream = await startWebcam();
    }
    if (!stream) return;

    setCallStatus("creating");
    const callDocRef = doc(collection(db, "calls"));
    const newCallId = callDocRef.id;

    initializePeerConnection(newCallId, true);
    if (!pc.current) return console.error("Peer connection not created");

    const offerCandidatesCol = collection(callDocRef, "offerCandidates");
    const answerCandidatesCol = collection(callDocRef, "answerCandidates");

    pc.current.onicecandidate = (event) => {
      if (event.candidate) addDoc(offerCandidatesCol, event.candidate.toJSON());
    };

    setupSignalingListeners(callDocRef, true);

    const unsubCandidates = onSnapshot(
      query(answerCandidatesCol),
      (snapshot: QuerySnapshot) => {
        snapshot.docChanges().forEach((change) => {
          if (change.type === "added") {
            pc.current?.addIceCandidate(new RTCIceCandidate(change.doc.data()));
          }
        });
      }
    );
    signalingUnsubscribers.current.push(unsubCandidates);

    const offerDescription = await pc.current.createOffer();
    await pc.current.setLocalDescription(offerDescription);
    const offer = { sdp: offerDescription.sdp, type: offerDescription.type };
    await setDoc(callDocRef, { offer });

    setCallId(newCallId);
    setCallStatus("waiting");
  }, [startWebcam, initializePeerConnection, setupSignalingListeners]);

  const joinCall = useCallback(
    async (joiningCallId: string) => {
      let stream = localStreamRef.current;
      if (!stream) {
        stream = await startWebcam();
      }
      if (!stream) return;

      setCallStatus("connecting");
      try {
        const callDocRef = doc(db, "calls", joiningCallId);
        const callDocSnap = await getDoc(callDocRef);
        if (!callDocSnap.exists()) {
          throw new Error("Call ID not found.");
        }

        setCallId(joiningCallId);
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
        const answer = {
          type: answerDescription.type,
          sdp: answerDescription.sdp,
        };
        await updateDoc(callDocRef, { answer, offer: deleteField() });

        const unsubCandidates = onSnapshot(
          query(offerCandidatesCol),
          (snapshot: QuerySnapshot) => {
            snapshot.docChanges().forEach((change) => {
              if (change.type === "added") {
                pc.current?.addIceCandidate(
                  new RTCIceCandidate(change.doc.data())
                );
              }
            });
          }
        );
        signalingUnsubscribers.current.push(unsubCandidates);
      } catch (err: any) {
        console.error(err);
        setError(err.message || "Failed to join call.");
        setCallStatus("error");
        await hangUp();
      }
    },
    [startWebcam, initializePeerConnection, setupSignalingListeners, hangUp]
  );

  const toggleMic = useCallback(() => {
    if (localStreamRef.current) {
      localStreamRef.current.getAudioTracks().forEach((track) => {
        track.enabled = !track.enabled;
      });
      setIsMuted((prev) => !prev);
    }
  }, []);

  const toggleVideo = useCallback(() => {
    if (localStreamRef.current) {
      localStreamRef.current.getVideoTracks().forEach((track) => {
        track.enabled = !track.enabled;
      });
      setIsVideoOff((prev) => !prev);
    }
  }, []);

  const toggleScreenShare = useCallback(async () => {
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
      } catch (err) {
        console.error("Error starting screen share:", err);
        setIsSharingScreen(false);
      }
    }
  }, [isSharingScreen]);

  useEffect(() => {
    return () => {
      if (pc.current) {
        hangUp();
      }
    };
  }, [hangUp]);

  return {
    localStream,
    remoteStream,
    callId,
    callStatus,
    error,
    isMuted,
    isVideoOff,
    isSharingScreen,
    startWebcam,
    createCall,
    joinCall,
    hangUp,
    toggleMic,
    toggleVideo,
    toggleScreenShare,
  };
};
