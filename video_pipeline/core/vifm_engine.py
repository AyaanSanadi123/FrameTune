# core/vifm_engine.py

import av
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from config.settings import VIFM_MODEL_ID, FRAMES_PER_CLIP

_model = None
_processor = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_vifm():
    """
    Loads the LanguageBind model into VRAM.
    Called exactly once by the orchestrator.
    """
    global _model, _processor
    
    print(f"Loading 3D Video Foundation Model on {_device}...")
    _processor = AutoProcessor.from_pretrained(VIFM_MODEL_ID)
    _model = AutoModel.from_pretrained(VIFM_MODEL_ID).to(_device)

def extract_tubelet(filepath: str) -> list:
    """
    Uses PyAV to extract exactly 8 uniformly spaced frames from the dynamic chunk.
    Fully compatible with Mac (Apple Silicon), Windows, and Linux.
    """
    container = av.open(filepath)
    stream = container.streams.video[0]
    
    total_frames = stream.frames
    
    if total_frames > 0:
        target_indices = set(np.linspace(0, total_frames - 1, FRAMES_PER_CLIP, dtype=int))
    else:
        target_indices = None 

    frames = []
    frame_count = 0
    
    for frame in container.decode(video=0):
        if target_indices is None or frame_count in target_indices:
            rgb_frame = frame.to_ndarray(format="rgb24")
            frames.append(rgb_frame)
            
        frame_count += 1
        if len(frames) == FRAMES_PER_CLIP:
            break
            
    container.close()
    
    if len(frames) != FRAMES_PER_CLIP:
        indices = np.linspace(0, len(frames) - 1, FRAMES_PER_CLIP, dtype=int)
        frames = [frames[i] for i in indices]
        
    return frames

def extract_video_embedding(filepath: str) -> torch.Tensor:
    """
    Embeds the 3D video tubelet and returns the RAW normalized spatiotemporal vector.
    Math and coordinate mapping are strictly delegated to coordinate_map.py.
    """
    if _model is None:
        raise RuntimeError("ViFM not initialized. Call initialize_vifm() first.")
        
    tubelet = extract_tubelet(filepath)
    
    # Processor handles the 3D stacking natively
    inputs = _processor(videos=tubelet, return_tensors="pt").to(_device)
    
    with torch.no_grad():
        video_features = _model.get_video_features(**inputs)
        # L2-normalize the vector before handing it off to the coordinate map
        video_embedding = F.normalize(video_features, p=2, dim=-1)
        
    return video_embedding