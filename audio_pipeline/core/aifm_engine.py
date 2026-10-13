# audio_pipeline/core/aifm_engine.py

import torch
import torch.nn.functional as F
from languagebind import LanguageBindAudio, LanguageBindAudioProcessor
from ..config.audio_settings import AIFM_MODEL_ID

_model = None
_processor = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_aifm():
    """Loads the Audio Foundation Model into global memory once."""
    global _model, _processor

    print(f"Loading Audio Foundation Model on {_device}...")

    _processor = LanguageBindAudioProcessor.from_pretrained(AIFM_MODEL_ID)

    # Standard GPU 16-bit precision for maximum CUDA acceleration, mirroring the video pipeline
    _model = LanguageBindAudio.from_pretrained(
        AIFM_MODEL_ID,
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32
    ).to(_device)

    _model.eval()
    print("AIFM model loaded successfully.")

def extract_audio_embedding(filepath: str) -> list[float]:
    """
    Embeds the 4-8s audio chunk and returns the RAW normalized spatiotemporal vector as a list.
    """
    if _model is None:
        raise RuntimeError("AIFM not initialized. Call initialize_aifm() first.")
        
    with torch.no_grad():
        # The processor handles the 16kHz resampling and Mel-spectrogram generation
        inputs = _processor(filepath, return_tensors="pt").to(_device)
        
        # Forward pass
        outputs = _model(**inputs)
        audio_features = outputs.audio_embeds
        
        # Explicit L2 normalization and float32 cast (matching ViFM)
        audio_embedding = F.normalize(audio_features.to(torch.float32), p=2, dim=-1, eps=1e-8)
        
        # Flatten for JSON/pgvector compatibility
        vector = audio_embedding[0].cpu().numpy().tolist()
        
    return vector