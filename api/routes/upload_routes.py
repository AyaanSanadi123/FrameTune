# api/routes/upload_routes.py

import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from config.database import supabase_client

router = APIRouter()

@router.post("/upload_video")
def upload_video(file: UploadFile = File(...)):
    """
    Phase 1: Secures the 4K video asset in cloud storage and creates a pending database row.
    """
    if not file.filename.endswith(('.mp4', '.mov')):
        raise HTTPException(status_code=400, detail="Only .mp4 and .mov files are supported.")

    video_id = str(uuid.uuid4())
    temp_filepath = f"data/raw/temp_{video_id}_{file.filename}"
    storage_path = f"raw_videos/{video_id}_{file.filename}" 

    try:
        # 1. Save locally to prevent server RAM bloat from massive files
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Upload the saved file to the Supabase Storage bucket
        with open(temp_filepath, "rb") as f:
            supabase_client.storage.from_("videos").upload(
                path=storage_path, 
                file=f,
                file_options={"content-type": file.content_type}
            )

        # 3. Create the ledger entry in PostgreSQL
        supabase_client.table("videos").insert({
            "id": video_id,
            "original_filename": file.filename,
            "storage_path": storage_path,
            "status": "pending"
        }).execute()

        return {
            "message": "Asset secured successfully.",
            "video_id": video_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cloud Storage Error: {str(e)}")

    finally:
        # 4. Wipe the local temp file to prevent storage exhaustion
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)