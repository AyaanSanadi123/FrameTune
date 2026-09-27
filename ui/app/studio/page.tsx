// src/app/studio/page.tsx

"use client";

import { useVideoPipeline } from "@/features/video/hooks/use-video-pipeline";
import { VideoDropzone } from "@/features/video/components/video-dropzone";
import { PipelineStatusIndicator } from "@/features/video/components/pipeline-status";
import { TrajectoryGrid } from "@/features/video/components/trajectory-grid";

export default function StudioPage() {
  // Extract videoId from the hook
  const { status, errorMessage, videoId, startPipeline, resetPipeline } = useVideoPipeline();

  return (
    <main className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <header className="mb-8 text-center sm:text-left">
          <h1 className="text-3xl font-extrabold text-gray-900 tracking-tight">
            Video Analysis Studio
          </h1>
          <p className="mt-2 text-sm text-gray-600">
            Secure cloud upload and machine-learning trajectory extraction.
          </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
          <section className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 min-h-[300px]">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">Input Video</h2>
            <VideoDropzone
              status={status}
              errorMessage={errorMessage}
              startPipeline={startPipeline}
              resetPipeline={resetPipeline}
            />
          </section>

          <section className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 min-h-[300px]">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">Execution Status</h2>
            
            {status === "idle" && (
              <div className="h-48 flex items-center justify-center text-sm text-gray-400 border border-dashed border-gray-200 rounded-lg">
                Waiting for video selection...
              </div>
            )}

            {/* Show progress tracker during upload/analysis */}
            {(status === "uploading" || status === "analyzing" || status === "error") && (
              <PipelineStatusIndicator status={status} />
            )}

            {/* Replace progress tracker with the fetched data once complete */}
            {status === "completed" && videoId && (
              <TrajectoryGrid videoId={videoId} />
            )}
          </section>
        </div>
      </div>
    </main>
  );
}