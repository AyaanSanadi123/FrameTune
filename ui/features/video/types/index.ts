// src/features/video/types/index.ts

// this is the response after video is uploaded 
export interface UploadResponse {
  message: string;
  video_id: string;
}

// this is the output from the /analyze endpoint
export interface AnalyzeResponse {
  message: string;
  video_id: string;
  status: string;
}