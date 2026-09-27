// src/features/video/components/pipeline-status.tsx

import { PipelineStatus } from "../hooks/use-video-pipeline";

interface PipelineStatusProps {
  status: PipelineStatus;
}

export function PipelineStatusIndicator({ status }: PipelineStatusProps) {
  if (status === "idle") return null;

  const isUploading = status === "uploading";
  const isAnalyzing = status === "analyzing";
  const isCompleted = status === "completed";
  const isError = status === "error";

  return (
    <div className="w-full max-w-md mx-auto mt-6 p-4 bg-white rounded-lg shadow-sm border border-gray-100">
      <h4 className="text-sm font-semibold text-gray-700 mb-4">Pipeline Status</h4>
      
      <div className="space-y-4">
        {/* Step 1: Upload */}
        <div className="flex items-center gap-3">
          <StatusIcon 
            isActive={isUploading} 
            isDone={isAnalyzing || isCompleted} 
            isError={isError && isUploading} 
          />
          <span className={`text-sm ${isUploading ? "text-blue-600 font-medium" : "text-gray-500"}`}>
            Phase 1: Securing Asset in Cloud Storage
          </span>
        </div>

        {/* Step 2: ML Analysis */}
        <div className="flex items-center gap-3">
          <StatusIcon 
            isActive={isAnalyzing} 
            isDone={isCompleted} 
            isError={isError && isAnalyzing} 
          />
          <span className={`text-sm ${isAnalyzing ? "text-blue-600 font-medium" : "text-gray-500"}`}>
            Phase 2: Running Machine Learning Orchestrator
          </span>
        </div>

        {/* Step 3: Database Write */}
        <div className="flex items-center gap-3">
          <StatusIcon 
            isActive={false} 
            isDone={isCompleted} 
            isError={false} 
          />
          <span className={`text-sm ${isCompleted ? "text-green-600 font-medium" : "text-gray-500"}`}>
            Phase 3: Trajectory Saved to Database
          </span>
        </div>
      </div>
    </div>
  );
}

// A small helper component to render the correct circle icon
function StatusIcon({ isActive, isDone, isError }: { isActive: boolean; isDone: boolean; isError: boolean }) {
  if (isError) {
    return <div className="w-5 h-5 rounded-full bg-red-500 flex-shrink-0" />;
  }
  if (isDone) {
    return (
      <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center flex-shrink-0">
        <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
        </svg>
      </div>
    );
  }
  if (isActive) {
    return (
      <div className="w-5 h-5 rounded-full border-2 border-blue-500 border-t-transparent animate-spin flex-shrink-0" />
    );
  }
  return <div className="w-5 h-5 rounded-full border-2 border-gray-200 flex-shrink-0" />;
}