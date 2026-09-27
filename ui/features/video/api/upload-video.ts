// src/features/video/api/upload-video.ts

import { apiClient } from "@/lib/api-client";
import { UploadResponse } from "../types";

export async function uploadVideo(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return apiClient<UploadResponse>("/upload_video", {
    method: "POST",
    body: formData,
  });
}