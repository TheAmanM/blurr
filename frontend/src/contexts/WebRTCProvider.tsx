import React, { createContext, useContext } from "react";
import { useWebRTC } from "../hooks/useWebRTC";

// Create a type for the context value for better TypeScript support
type WebRTCContextType = ReturnType<typeof useWebRTC>;

const WebRTCContext = createContext<WebRTCContextType | null>(null);

export const WebRTCProvider = ({ children }: { children: React.ReactNode }) => {
  const webRTC = useWebRTC();
  return (
    <WebRTCContext.Provider value={webRTC}>{children}</WebRTCContext.Provider>
  );
};

// Custom hook to easily consume the context
export const useWebRTCContext = () => {
  const context = useContext(WebRTCContext);
  if (!context) {
    throw new Error("useWebRTCContext must be used within a WebRTCProvider");
  }
  return context;
};
