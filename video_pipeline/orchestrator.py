
# video_pipeline/orchestrator.py

import os
import av
import torch
from .core.segmentation import segment_video
from .core.vifm_engine import extract_video_embedding

def _get_chunk_duration(filepath: str) -> float:
    """Helper to get the exact duration of a chunk in seconds using PyAV."""
    with av.open(filepath) as container:
        return float(container.duration) / av.time_base

def process_video_trajectory(raw_video_path: str, coord_space_instance) -> list:
    """
    Main ML Pipeline:
    1. Slices video into cinematic action segments via TransNetV2.
    2. Extracts dense spatiotemporal embeddings via LanguageBind.
    3. Projects embeddings to 2D coordinates for UI visualization.
    4. Packages chunks with vector embeddings for PostgreSQL / pgvector insertion.
    """
    print(f"[Orchestrator] Starting analysis on: {raw_video_path}")
    
    chunks_dir = "data/processed/chunks"
    os.makedirs(chunks_dir, exist_ok=True)
    
    # 1. Dynamic Shot Segmentation
    chunk_filepaths = segment_video(raw_video_path, chunks_dir)
    
    if not chunk_filepaths:
        print("[Orchestrator] Error: No cinematic shots detected.")
        return []

    trajectory = []
    current_timestamp = 0.0
    
    print("[Orchestrator] Extracting spatiotemporal embeddings & generating vector matrix...")
    
    # 2. Sequential Processing Loop
    for idx, chunk_path in enumerate(chunk_filepaths):
        try:
            duration = _get_chunk_duration(chunk_path)
            
            # Extract raw 3D tensor from LanguageBind
            raw_embedding = extract_video_embedding(chunk_path)
            
            # Flatten & convert tensor to 1D Python float list for pgvector
            if isinstance(raw_embedding, torch.Tensor):
                vector_payload = raw_embedding.detach().cpu().flatten().tolist()
            else:
                vector_payload = list(raw_embedding)
            
            # Calculate 2D anchor projection for UI
            coords = coord_space_instance.project_embedding(raw_embedding)
            
            # Construct chunk metadata record matching the database schema
            trajectory.append({
                "chunk_order": idx,
                "start_time": round(current_timestamp, 2),
                "duration": round(duration, 2),
                "ui_coordinate": {
                    "mood": round(coords["mood"], 4),
                    "energy": round(coords["energy"], 4)
                },
                "embedding": vector_payload
            })
            
            print(f"   -> Chunk {idx:03d} | Length: {duration:.2f}s | Mood: {coords['mood']:.2f}, Energy: {coords['energy']:.2f} | Vector Dim: {len(vector_payload)}")
            current_timestamp += duration
            
        except Exception as e:
            print(f"[Orchestrator] Failed on {chunk_path}: {e}")
            continue
            
        finally:
            # Clean up the local physical chunk file to prevent disk exhaustion
            if os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                except OSError:
                    pass

    print(f"[Orchestrator] Pipeline complete. Processed {len(trajectory)} chunks.")
    return trajectory