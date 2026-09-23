# core/segmentation.py

import os
import cv2
import ffmpeg
import numpy as np
from transnetv2pt import predict_video
from config.settings import TARGET_FPS

def segment_video(input_filepath: str, output_dir: str) -> list:
    """
    Analyzes the video for cinematic cuts and slices it into dynamic chunks.
    Merges micro-shots (under 1s) to preserve rapid-fire action montages.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Clear old chunks
    for old_chunk in os.listdir(output_dir):
        if old_chunk.startswith("chunk_") and old_chunk.endswith(".mp4"):
            os.remove(os.path.join(output_dir, old_chunk))

    # 2. Run TransNetV2
    print("Detecting cinematic shots with TransNetV2...")
    raw_scenes = predict_video(input_filepath)
    
    # 3. The Merging Algorithm
    merged_scenes = []
    for start_frame, end_frame in raw_scenes:
        if not merged_scenes:
            merged_scenes.append([start_frame, end_frame])
            continue
            
        current_duration = (end_frame - start_frame) / TARGET_FPS
        
        # If the current shot is a micro-shot, merge it into the previous chunk
        if current_duration < 1.0:
            merged_scenes[-1][1] = end_frame
        else:
            # If the previous chunk was a micro-shot (e.g., the very first scene), 
            # merge it into this valid chunk instead.
            prev_duration = (merged_scenes[-1][1] - merged_scenes[-1][0]) / TARGET_FPS
            if prev_duration < 1.0:
                merged_scenes[-1][1] = end_frame
            else:
                merged_scenes.append([start_frame, end_frame])
    
    chunk_filepaths = []
    
    # 4. Physically slice the merged chunks
    for i, (start_frame, end_frame) in enumerate(merged_scenes):
        start_time = float(start_frame) / TARGET_FPS
        duration = float(end_frame - start_frame) / TARGET_FPS
            
        output_filepath = os.path.join(output_dir, f"chunk_{i:03d}.mp4")
        
        try:
            (
                ffmpeg
                .input(input_filepath, ss=start_time, t=duration)
                .output(
                    output_filepath,
                    vcodec='libx264',
                    an=None,
                    reset_timestamps=1,
                    strict='experimental'
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            chunk_filepaths.append(output_filepath)
            
        except ffmpeg.Error as e:
            print(f"FFmpeg Error on chunk {i}:", e.stderr.decode('utf8'))
            continue
            
    print(f"TransNetV2 compiled {len(chunk_filepaths)} continuous cinematic sequences.")
    return chunk_filepaths