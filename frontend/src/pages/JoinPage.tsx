import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import { Label } from "../components/ui/label";
import Background from "../components/Background";
import { useWebRTCContext } from "../contexts/WebRTCProvider";

export default function JoinPage() {
  const [meetingCode, setMeetingCode] = useState("");
  const navigate = useNavigate();
  const { startWebcam } = useWebRTCContext();

  const handleJoin = async () => {
    if (!meetingCode.trim()) {
      alert("Please enter a meeting code.");
      return;
    }
    // Pre-start webcam to request permissions early
    await startWebcam();
    navigate(`/join/${meetingCode.trim()}`);
  };

  const handleCreate = async () => {
    // Pre-start webcam to request permissions early
    await startWebcam();
    navigate("/create");
  };

  return (
    <Background>
      <Card className="w-full max-w-md mx-4">
        <CardHeader>
          <CardTitle className="text-2xl lg:text-3xl">Video Call</CardTitle>
          <CardDescription className="text-muted-foreground lg:text-base">
            Enter a call ID to join or create a new call.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="meeting-code">Meeting Code</Label>
            <Input
              id="meeting-code"
              placeholder="Enter Call ID"
              value={meetingCode}
              onChange={(e) => setMeetingCode(e.target.value)}
              onKeyUp={(e) => e.key === "Enter" && handleJoin()}
            />
          </div>
          <Button
            onClick={handleJoin}
            className="w-full"
            disabled={!meetingCode.trim()}
          >
            Join Call
          </Button>
        </CardContent>
        <CardFooter className="flex flex-col items-center">
          <div className="relative w-full flex items-center mb-6">
            <div className="flex-grow border-t border-[#222]"></div>
            <span className="flex-shrink mx-4 text-xs text-[#333] uppercase">
              Or
            </span>
            <div className="flex-grow border-t border-[#222]"></div>
          </div>
          <Button onClick={handleCreate} variant="outline" className="w-full">
            Create a New Call
          </Button>
        </CardFooter>
      </Card>
    </Background>
  );
}
