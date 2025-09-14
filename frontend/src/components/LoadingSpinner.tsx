import React from "react";

interface LoadingSpinnerProps {
  text?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ text }) => {
  return (
    <div
      className="flex flex-col items-center justify-center h-full text-center"
      role="status"
      aria-live="polite"
    >
      <svg
        className="animate-spin duration-1200 h-12 w-12 text-white"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 50 50"
      >
        <circle
          cx="25"
          cy="25"
          r="20"
          stroke="currentColor"
          strokeWidth="5"
          strokeLinecap="round"
          strokeDasharray="31.4 31.4" /* This creates two 90-degree arcs on opposite sides */
        />
      </svg>
      {text && (
        <p className="mt-4 text-lg text-gray-300 animate-pulse">{text}</p>
      )}
      <span className="sr-only">Loading...</span>
    </div>
  );
};

export default LoadingSpinner;
