# core/vifm_engine.py

import torch
import decord
import numpy as np
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel

from config.settings import (
    VIFM_MODEL_ID, 
    FRAMES_PER_CLIP, 
    T_ENERGY_HIGH, T_ENERGY_LOW, 
    T_MOOD_BRIGHT, T_MOOD_DARK
)

_model = None
_processor = None
_anchors = {}
_device = "cuda" if torch.cuda.is_available() else "cpu"

def _embed_and_normalize_text(prompts: list) -> torch.Tensor:
    """Helper function to tokenize, embed, average, and normalize text anchors."""
    inputs = _processor(text=prompts, return_tensors="pt", padding=True).to(_device)
    with torch.no_grad():
        # Get text features and average them into a single "super anchor"
        features = _model.get_text_features(**inputs)
        mean_feature = features.mean(dim=0, keepdim=True)
        return F.normalize(mean_feature, p=2, dim=-1)

def initialize_vifm():
    """
    Loads the LanguageBind model into VRAM and pre-calculates both coordinate axes.
    Called exactly once by the orchestrator.
    """
    global _model, _processor, _anchors
    
    print(f"Loading 3D Video Foundation Model on {_device}...")
    _processor = AutoProcessor.from_pretrained(VIFM_MODEL_ID)
    _model = AutoModel.from_pretrained(VIFM_MODEL_ID).to(_device)
    
    # Pre-calculate the absolute poles of our 2D coordinate plane
    _anchors["energy_high"] = _embed_and_normalize_text(T_ENERGY_HIGH)
    _anchors["energy_low"] = _embed_and_normalize_text(T_ENERGY_LOW)
    _anchors["mood_bright"] = _embed_and_normalize_text(T_MOOD_BRIGHT)
    _anchors["mood_dark"] = _embed_and_normalize_text(T_MOOD_DARK)

def extract_tubelet(filepath: str) -> list:
    """
    Uses Decord to extract exactly 8 uniformly spaced frames from the dynamic chunk.
    """
    vr = decord.VideoReader(filepath, num_threads=1, ctx=decord.cpu(0))
    total_frames = len(vr)
    
    # Generate 8 perfectly spaced frame indices across the duration of the shot
    indices = np.linspace(0, total_frames - 1, FRAMES_PER_CLIP, dtype=int)
    
    # Extract just those 8 frames and convert them to standard numpy arrays
    frames = vr.get_batch(indices).asnumpy()
    return list(frames)

def process_spatiotemporal_metrics(filepath: str) -> dict:
    """
    Embeds the 3D video tubelet and calculates the final [Energy, Mood] coordinate.
    """
    if _model is None:
        raise RuntimeError("ViFM not initialized. Call initialize_vifm() first.")
        
    tubelet = extract_tubelet(filepath)
    
    # The processor natively handles the complex 3D stacking (Frames x Channels x Height x Width)
    inputs = _processor(videos=tubelet, return_tensors="pt").to(_device)
    
    with torch.no_grad():
        # 1. The Single Heavy Forward Pass
        video_features = _model.get_video_features(**inputs)
        video_embedding = F.normalize(video_features, p=2, dim=-1)
        
        # 2. Axis 1: Energy Projection
        sim_e_low = torch.matmul(video_embedding, _anchors["energy_low"].T).squeeze()
        sim_e_high = torch.matmul(video_embedding, _anchors["energy_high"].T).squeeze()
        
        energy_probs = F.softmax(torch.stack([sim_e_low, sim_e_high]), dim=0)
        final_energy = float(energy_probs[1].item())
        
        # 3. Axis 2: Mood Projection
        sim_m_dark = torch.matmul(video_embedding, _anchors["mood_dark"].T).squeeze()
        sim_m_bright = torch.matmul(video_embedding, _anchors["mood_bright"].T).squeeze()
        
        mood_probs = F.softmax(torch.stack([sim_m_dark, sim_m_bright]), dim=0)
        final_mood = float(mood_probs[1].item())
        
    return {
        "energy": final_energy,
        "mood": final_mood
    }