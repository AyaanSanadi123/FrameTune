// src/features/video/hooks/use-video-pipeline.ts

import { useState } from "react";
import { uploadVideo } from "../api/upload-video";
import { triggerAnalysis } from "../api/trigger-analysis";

export type PipelineStatus = "idle" | "uploading" | "analyzing" | "completed" | "error";

export function useVideoPipeline() {
  const [status, setStatus] = useState<PipelineStatus>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);

  const startPipeline = async (file: File) => {
    setStatus("uploading");
    setErrorMessage(null);

    try {
      // Phase 1: Secure the asset in the cloud
      const uploadRes = await uploadVideo(file);
      setVideoId(uploadRes.video_id);

      // Phase 2: Trigger the ML orchestrator
      setStatus("analyzing");
      await triggerAnalysis(uploadRes.video_id);

      // Success: The database now holds the final trajectory
      setStatus("completed");

    } catch (error: any) {
      setStatus("error");
      setErrorMessage(error.message || "An unexpected error occurred in the pipeline.");
    }
  };

  const resetPipeline = () => {
    setStatus("idle");
    setErrorMessage(null);
    setVideoId(null);
  };

  return {
    status,
    errorMessage,
    videoId,
    startPipeline,
    resetPipeline
  };
}