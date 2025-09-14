import React from "react";
import { Link } from "react-router-dom";
import Background from "./Background";

interface ErrorDisplayProps {
  message: string | null;
  onRetry?: () => void;
}

const ErrorDisplay: React.FC<ErrorDisplayProps> = ({ message, onRetry }) => {
  if (!message) {
    return null;
  }

  return (
    <Background>
      <div
        className="flex flex-col items-center justify-center text-center bg-red-900/50 bg-opacity-20 p-8 rounded-lg"
        role="alert"
      >
        <div className="w-16 h-16 flex items-center justify-center bg-red-500 rounded-full mb-4">
          <svg
            className="w-8 h-8 text-white"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            ></path>
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-red-400 mb-2">
          An Error Occurred
        </h2>
        <p className="text-gray-300 mb-6">{message}</p>
        <div className="flex items-center gap-4">
          <Link
            to="/join"
            className="px-6 py-2 bg-gray-600 hover:bg-gray-500 rounded-lg font-semibold transition-colors"
          >
            Go Home
          </Link>
          {onRetry && (
            <button
              onClick={onRetry}
              className="px-6 py-2 bg-teal-600 hover:bg-teal-700 rounded-lg font-semibold transition-colors"
            >
              Try Again
            </button>
          )}
        </div>
      </div>
    </Background>
  );
};

export default ErrorDisplay;
