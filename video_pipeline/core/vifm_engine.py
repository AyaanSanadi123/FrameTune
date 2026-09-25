# core/vifm_engine.py

import av
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from languagebind import (
    LanguageBindVideo,
    LanguageBindVideoTokenizer,
    LanguageBindVideoProcessor,
)
from languagebind.video.configuration_video import LanguageBindVideoConfig
from video_pipeline.config.settings import VIFM_MODEL_ID, FRAMES_PER_CLIP

_model = None
_processor = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_vifm():
    global _model, _processor

    print(f"Loading 3D Video Foundation Model on {_device}...")

    config = LanguageBindVideoConfig.from_pretrained(VIFM_MODEL_ID)
    tokenizer = LanguageBindVideoTokenizer.from_pretrained(VIFM_MODEL_ID)

    _processor = LanguageBindVideoProcessor(
        config=config,
        tokenizer=tokenizer,
    )

    # 1. The bfloat16 Fix: Same memory as 16-bit, but the massive numerical range of 32-bit
    _model = LanguageBindVideo.from_pretrained(
        VIFM_MODEL_ID,
        config=config,
        torch_dtype=torch.bfloat16 
    ).to(_device)

    # 2. The Attention Fix: Commented out to allow PyTorch to use optimized SDPA
    # _model.config._attn_implementation = "eager"
    # _model.text_model.config._attn_implementation = "eager"
    # _model.vision_model.config._attn_implementation = "eager"

    _model.eval()
    print("ViFM model loaded successfully.")

def extract_tubelet(filepath: str) -> list:
    """
    Uses PyAV to extract exactly 8 uniformly spaced frames from the dynamic chunk.
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
            pil_frame = Image.fromarray(rgb_frame)
            frames.append(pil_frame)
            
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
    Bypasses the LanguageBindProcessor to natively support Apple Silicon & PyAV.
    """
    if _model is None:
        raise RuntimeError("ViFM not initialized. Call initialize_vifm() first.")
        
    tubelet = extract_tubelet(filepath)
    
    clip_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    tensor_frames = [clip_transform(frame) for frame in tubelet]
    video_tensor = torch.stack(tensor_frames, dim=1)
    
    # 3. Dynamically match the input tensor to the model's bfloat16 datatype
    pixel_values = video_tensor.unsqueeze(0).to(dtype=_model.dtype, device=_device)
    
    # Pass a valid sentence instead of an empty string to prevent text encoder collapse
    dummy_text = _processor.tokenizer(["a cinematic video clip"], return_tensors="pt")
    input_ids = dummy_text.input_ids.to(_device)
    
    with torch.no_grad():
        outputs = _model(input_ids=input_ids, pixel_values=pixel_values)
        video_features = outputs.image_embeds
        
        # Safety Net: If the CPU still overflows, log it and mathematically sanitize NaNs
        if torch.isnan(video_features).any():
            print("[Warning] Overflow detected. Vector zeroed.")
            video_features = torch.nan_to_num(video_features, nan=0.0)
            
        # Cast back to standard float32 for the JSON parser at the very end
        video_features = video_features.to(torch.float32)
        video_embedding = F.normalize(video_features, p=2, dim=-1, eps=1e-8)
        
    return video_embedding