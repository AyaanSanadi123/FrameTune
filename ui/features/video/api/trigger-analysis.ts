// src/features/video/api/trigger-analysis.ts

import { apiClient } from "@/lib/api-client";
import { AnalyzeResponse } from "../types";

export async function triggerAnalysis(videoId: string): Promise<AnalyzeResponse> {
  return apiClient<AnalyzeResponse>("/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ video_id: videoId }),
  });
}