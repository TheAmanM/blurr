import { useEffect } from "react";
import { useParams } from "react-router-dom";
import { useWebRTCContext } from "../contexts/WebRTCProvider";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorDisplay from "../components/ErrorDisplay";
import CallPage from "./CallPage";
import Background from "../components/Background";

const JoinByIdPage = () => {
  const { id } = useParams<{ id: string }>();
  const { callStatus, error, joinCall } = useWebRTCContext();

  useEffect(() => {
    if (id) {
      joinCall(id);
    }
  }, [joinCall, id]);

  if (callStatus === "error") {
    return <ErrorDisplay message={error} />;
  }

  if (callStatus === "connected") {
    return <CallPage />;
  }

  return (
    <Background>
      <LoadingSpinner text="Joining call..." />
    </Background>
  );
};

export default JoinByIdPage;
