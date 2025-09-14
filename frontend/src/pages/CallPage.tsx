import { useWebRTCContext } from "../contexts/WebRTCProvider";
import VideoPlayer from "../components/VideoPlayer";
import CallControls from "../components/CallControls";
import Background from "../components/Background";

const CallPage = () => {
  const { localStream, remoteStream } = useWebRTCContext();

  return (
    <Background>
      <div className="flex-grow flex flex-col items-center justify-center">
        <div className="grid md:grid-cols-2 gap-4 self-center">
          <VideoPlayer stream={localStream} isMuted={true} />
          <VideoPlayer stream={remoteStream} />
        </div>
        <div className="flex-shrink-0 mt-4">
          <CallControls />
        </div>
      </div>
    </Background>
  );
};

export default CallPage;
