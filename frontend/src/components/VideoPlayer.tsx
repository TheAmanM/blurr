import React, { useEffect, useRef } from "react";

interface VideoPlayerProps {
  stream: MediaStream | null;
  isMuted?: boolean;
  label?: string;
  placeholderText?: string;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({
  stream,
  isMuted = false,
  label,
  placeholderText = "Waiting for connection...",
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  return (
    <div className="relative bg-black rounded-lg overflow-hidden shadow-sm w-full aspect-video flex items-center justify-center">
      {stream ? (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted={isMuted}
          className="w-full h-full object-contain"
        />
      ) : (
        <p className="text-gray-400">{placeholderText}</p>
      )}
      {label && (
        <span className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white text-sm px-2 py-1 rounded">
          {label}
        </span>
      )}
    </div>
  );
};

export default VideoPlayer;
