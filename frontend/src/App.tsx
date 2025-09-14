import { Outlet } from "react-router-dom";
import { WebRTCProvider } from "./contexts/WebRTCProvider";

function App() {
  return (
    <WebRTCProvider>
      <div className="bg-gray-900 text-white min-h-screen flex flex-col items-center font-sans">
        <Outlet /> {/* This will render the matched route component */}
      </div>
    </WebRTCProvider>
  );
}

export default App;
