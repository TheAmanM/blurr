import { useState } from "react";
import { useWebRTCContext } from "../contexts/WebRTCProvider";

const MicOnIcon = () => (
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
);

const MicOffIcon = () => (
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
    <path strokeLinecap="round" strokeLinejoin="round" d="M5 5l14 14"></path>
  </svg>
);

const VideoOnIcon = () => (
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
);

const VideoOffIcon = () => (
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
    <path strokeLinecap="round" strokeLinejoin="round" d="M1 1l22 22"></path>
  </svg>
);

const ScreenShareIcon = () => (
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
);

const HangUpIcon = () => (
  <svg className="w-6 h-6" fill="currentColor" viewBox="0 -256 1792 1792">
    <g transform="matrix(1,0,0,-1,159.45763,1293.0169)">
      <path
        d="m 1408,296 q 0,-27 -10,-70.5 Q 1388,182 1377,157 1356,107 1255,51 1161,0 1069,0 1042,0 1016.5,3.5 991,7 959,16 927,25 911.5,30.5 896,36 856,51 816,66 807,69 709,104 632,152 504,231 367.5,367.5 231,504 152,632 104,709 69,807 66,816 51,856 36,896 30.5,911.5 25,927 16,959 7,991 3.5,1016.5 0,1042 0,1069 q 0,92 51,186 56,101 106,122 25,11 68.5,21 43.5,10 70.5,10 14,0 21,-3 18,-6 53,-76 11,-19 30,-54 19,-35 35,-63.5 16,-28.5 31,-53.5 3,-4 17.5,-25 14.5,-21 21.5,-35.5 7,-14.5 7,-28.5 0,-20 -28.5,-50 -28.5,-30 -62,-55 -33.5,-25 -62,-53 -28.5,-28 -28.5,-46 0,-9 5,-22.5 5,-13.5 8.5,-20.5 3.5,-7 14,-24 10.5,-17 11.5,-19 76,-137 174,-235 98,-98 235,-174 2,-1 19,-11.5 17,-10.5 24,-14 7,-3.5 20.5,-8.5 13.5,-5 22.5,-5 18,0 46,28.5 28,28.5 53,62 25,33.5 55,62 30,28.5 50,28.5 14,0 28.5,-7 14.5,-7 35.5,-21.5 21,-14.5 25,-17.5 25,-15 53.5,-31 28.5,-16 63.5,-35 35,-19 54,-30 70,-35 76,-53 3,-7 3,-21 z"
        style={{ fill: "currentColor" }}
      />
    </g>
  </svg>
);

// --- Main Component ---

const CallControls = () => {
  const {
    isMuted,
    isVideoOff,
    isSharingScreen,
    toggleMic,
    toggleVideo,
    toggleScreenShare,
    hangUp,
  } = useWebRTCContext();
  const [showEndCallConfirm, setShowEndCallConfirm] = useState(false);

  return (
    <>
      <div className="bg-[#FFFFFF04] backdrop-blur-[2px] p-4 rounded-lg shadow-2xl flex justify-center items-center gap-4">
        <button
          onClick={toggleMic}
          className={`p-4 rounded-full transition-colors text-white ${
            isMuted
              ? "bg-red-600 hover:bg-red-700"
              : "bg-gray-600 hover:bg-gray-500"
          }`}
          aria-label={isMuted ? "Unmute Microphone" : "Mute Microphone"}
        >
          {isMuted ? <MicOffIcon /> : <MicOnIcon />}
        </button>

        <button
          onClick={toggleVideo}
          className={`p-4 rounded-full transition-colors text-white ${
            isVideoOff
              ? "bg-red-600 hover:bg-red-700"
              : "bg-gray-600 hover:bg-gray-500"
          }`}
          aria-label={isVideoOff ? "Turn Camera On" : "Turn Camera Off"}
        >
          {isVideoOff ? <VideoOffIcon /> : <VideoOnIcon />}
        </button>

        <button
          onClick={toggleScreenShare}
          className={`p-4 rounded-full transition-colors text-white ${
            isSharingScreen
              ? "bg-blue-600 hover:bg-blue-700"
              : "bg-gray-600 hover:bg-gray-500"
          }`}
          aria-label={isSharingScreen ? "Stop Sharing Screen" : "Share Screen"}
        >
          <ScreenShareIcon />
        </button>

        <button
          onClick={() => setShowEndCallConfirm(true)}
          className="p-4 rounded-full bg-red-600 hover:bg-red-700 text-white"
          aria-label="End Call"
        >
          <HangUpIcon />
        </button>
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
    </>
  );
};

export default CallControls;
