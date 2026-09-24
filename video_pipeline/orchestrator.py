# video_pipeline/orchestrator.py

import os
import av
from .core.segmentation import segment_video
from .core.vifm_engine import extract_video_embedding

def _get_chunk_duration(filepath: str) -> float:
    """Helper to get the exact duration of a chunk in seconds using PyAV."""
    with av.open(filepath) as container:
        return float(container.duration) / av.time_base

def process_video_trajectory(raw_video_path: str, coord_space_instance) -> list:
    """
    The main ML pipeline. 
    Accepts a filepath and a pre-loaded UniversalCoordinateSpace instance.
    Returns a list of dictionaries mapping time to [Mood, Energy].
    """
    print(f"[Orchestrator] Starting analysis on: {raw_video_path}")
    
    chunks_dir = "data/processed/chunks"
    os.makedirs(chunks_dir, exist_ok=True)
    
    # 1. Dynamic Shot Segmentation
    # Slices the video into action-bound chunks
    chunk_filepaths = segment_video(raw_video_path, chunks_dir)
    
    if not chunk_filepaths:
        print("[Orchestrator] Error: No cinematic shots detected.")
        return []

    trajectory = []
    current_timestamp = 0.0
    
    print("[Orchestrator] Extracting spatiotemporal embeddings...")
    
    # 2. Sequential Processing Loop
    for idx, chunk_path in enumerate(chunk_filepaths):
        try:
            duration = _get_chunk_duration(chunk_path)
            
            # Extract raw 3D tensor from LanguageBind
            raw_embedding = extract_video_embedding(chunk_path)
            
            # Pass the tensor to the pre-loaded coordinate map for anchoring
            coords = coord_space_instance.project_embedding(raw_embedding)
            
            # Build the data point
            trajectory.append({
                "chunk_id": idx,
                "timestamp": round(current_timestamp, 2),
                "duration": round(duration, 2),
                "coordinate": [round(coords["mood"], 4), round(coords["energy"], 4)]
            })
            
            print(f"   -> Chunk {idx:03d} mapped to Mood: {coords['mood']:.2f}, Energy: {coords['energy']:.2f}")
            current_timestamp += duration
            
        except Exception as e:
            print(f"[Orchestrator] Failed on {chunk_path}: {e}")
            continue

    print("[Orchestrator] Pipeline complete.")
    return trajectory