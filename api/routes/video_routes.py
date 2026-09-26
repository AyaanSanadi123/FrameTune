# api/routes/video_routes.py

import os
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from config.database import supabase_client
from video_pipeline.orchestrator import process_video_trajectory

router = APIRouter()

class AnalyzeRequest(BaseModel):
    video_id: str

@router.post("/analyze")
def analyze_video(request: Request, payload: AnalyzeRequest):
    """
    Phase 2: Downloads the secured video from cloud storage, runs the ML pipeline, 
    and saves the trajectory to the database (Single Source of Truth).
    """
    video_id = payload.video_id
    temp_filepath = f"data/raw/temp_download_{video_id}.mp4"

    try:
        # 1. Query PostgreSQL for the file's exact location in the bucket
        db_response = supabase_client.table("videos").select("storage_path").eq("id", video_id).execute()
        if not db_response.data:
            raise HTTPException(status_code=404, detail="Video record not found.")
        
        storage_path = db_response.data[0]["storage_path"]

        # 2. Download the file from Supabase Storage to the local server
        with open(temp_filepath, "wb") as f:
            file_data = supabase_client.storage.from_("videos").download(storage_path)
            f.write(file_data)

        # 3. Retrieve the warm ML object from the server's global state
        warm_coord_space = getattr(request.app.state, "coord_space", None)
        if not warm_coord_space:
            raise HTTPException(status_code=503, detail="Machine learning models are booting.")

        # 4. Execute the stateless ML orchestrator
        trajectory = process_video_trajectory(temp_filepath, warm_coord_space)

        if not trajectory:
            raise HTTPException(status_code=422, detail="Pipeline failed to extract cinematic shots.")

        # 5. Write strictly to the Database
        supabase_client.table("videos").update({
            "status": "completed",
            "trajectory": trajectory
        }).eq("id", video_id).execute()

        return {
            "message": "Analysis complete. Trajectory saved to database.",
            "video_id": video_id,
            "status": "completed"
        }

    except Exception as e:
        # If the pipeline crashes, flag the row as failed so the frontend knows
        supabase_client.table("videos").update({"status": "failed"}).eq("id", video_id).execute()
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")

    finally:
        # 6. Always wipe the downloaded file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)