// src/features/video/components/video-dropzone.tsx

"use client";

import { useState, useRef } from "react";
import { PipelineStatus } from "../hooks/use-video-pipeline";

interface VideoDropzoneProps {
  status: PipelineStatus;
  errorMessage: string | null;
  startPipeline: (file: File) => Promise<void>;
  resetPipeline: () => void;
}

export function VideoDropzone({
  status,
  errorMessage,
  startPipeline,
  resetPipeline,
}: VideoDropzoneProps) {
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = () => {
    if (file) startPipeline(file);
  };

  const handleReset = () => {
    setFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    resetPipeline();
  };

  if (status === "completed") {
    return (
      <div className="p-6 border-2 border-green-500 rounded-lg text-center bg-green-50 h-full flex flex-col justify-center items-center">
        <h3 className="text-xl font-bold text-green-700 mb-2">Analysis Complete</h3>
        <p className="text-green-600 mb-4">The trajectory data has been securely saved to the database.</p>
        <button 
          onClick={handleReset}
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition"
        >
          Process Another Video
        </button>
      </div>
    );
  }

  return (
    <div className="p-6 border-2 border-dashed border-gray-300 rounded-lg bg-gray-50 h-full flex flex-col justify-between">
      <div className="flex flex-col gap-4">
        <label className="text-sm font-medium text-gray-700">
          Select Video (.mp4 or .mov)
        </label>
        
        <input
          type="file"
          accept="video/mp4, video/quicktime"
          onChange={handleFileChange}
          ref={fileInputRef}
          disabled={status === "uploading" || status === "analyzing"}
          className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50 cursor-pointer"
        />

        {status === "error" && (
          <div className="mt-2 p-3 bg-red-100 border border-red-400 text-red-700 rounded text-sm">
            <p className="font-bold">Pipeline Error</p>
            <p>{errorMessage}</p>
            <button onClick={handleReset} className="underline mt-1 text-red-800">Try Again</button>
          </div>
        )}
      </div>

      <button
        onClick={handleUpload}
        disabled={!file || status === "uploading" || status === "analyzing"}
        className="mt-6 w-full py-2.5 bg-blue-600 text-white font-medium rounded hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
      >
        {status === "idle" && "Upload & Analyze"}
        {status === "uploading" && "1/2: Securing Asset in Cloud..."}
        {status === "analyzing" && "2/2: Running ML Inference..."}
      </button>
    </div>
  );
}