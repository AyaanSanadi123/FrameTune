# api/routes/video_routes.py

import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from video_pipeline.orchestrator import process_video_trajectory

# Create the plug-and-play router module
router = APIRouter()

# Ensure standard working directories exist at runtime
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed/chunks", exist_ok=True)

@router.post("/analyze")
def analyze_video(request: Request, file: UploadFile = File(...)):
    """
    Receives a video file, runs it through the spatiotemporal ML pipeline,
    and returns the emotional trajectory mapped to the 2D coordinate grid.
    """
    # Reject non-video files immediately
    if not file.filename.endswith(('.mp4', '.mov')):
        raise HTTPException(status_code=400, detail="Only .mp4 and .mov files are supported.")

    # 1. Thread Safety: Append a short UUID so concurrent users don't overwrite each other's files
    file_id = str(uuid.uuid4())[:8]
    temp_filepath = f"data/raw/temp_{file_id}_{file.filename}"

    try:
        # 2. Save the incoming network stream directly to the hard drive
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Retrieve the warm ML object from the server's global state
        warm_coord_space = getattr(request.app.state, "coord_space", None)
        if not warm_coord_space:
            raise HTTPException(
                status_code=503, 
                detail="Machine learning models are not warmed up. Server is still booting."
            )

        # 4. Execution: Hand the file to the stateless orchestrator
        trajectory = process_video_trajectory(temp_filepath, warm_coord_space)

        if not trajectory:
            raise HTTPException(
                status_code=422, 
                detail="Pipeline failed to extract valid cinematic shots from this video."
            )

        # FastAPI automatically converts this dictionary into your final JSON payload
        return {
            "filename": file.filename,
            "trajectory": trajectory
        }

    except Exception as e:
        # Catch unexpected ML crashes and return a clean HTTP 500 to the frontend
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")

    finally:
        # 5. Cleanup: Always delete the raw video file to protect server storage,
        # even if the orchestrator crashed halfway through.
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)