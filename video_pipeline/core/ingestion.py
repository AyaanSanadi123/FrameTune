# core/ingestion.py

import os
import glob
import ffmpeg
from config.settings import TARGET_RESOLUTION, TARGET_FPS, CHUNK_DURATION_SEC

def process_video(input_filepath: str, output_dir: str) -> list:
    """
    Normalizes the raw video to target FPS and resolution, 
    and segments it into uniform chunks for parallel processing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear out any old chunks from previous runs to prevent cross-contamination
    for old_chunk in glob.glob(os.path.join(output_dir, "chunk_*.mp4")):
        os.remove(old_chunk)

    # %03d formats the output files as chunk_000.mp4, chunk_001.mp4, etc.
    output_pattern = os.path.join(output_dir, "chunk_%03d.mp4")

    try:
        # 1. Ingest the raw media file
        stream = ffmpeg.input(input_filepath)
        
        # 2. Normalize the visual track using FFmpeg filters
        # Scale to 640x480 and strictly enforce 30 FPS to match our heuristics
        video = stream.video.filter('scale', TARGET_RESOLUTION[0], TARGET_RESOLUTION[1])
        video = video.filter('fps', fps=TARGET_FPS, round='up')
        
        
        # 3. Execute the segment muxer
        (
            ffmpeg
            .output(
                video, 
                output_pattern,
                f='segment',
                segment_time=CHUNK_DURATION_SEC,
                reset_timestamps=1,
                vcodec='libx264',
                an=None,  # CRITICAL: -an flag strips the audio track to prevent crashes
                strict='experimental'
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        # Decode and print the exact FFmpeg binary error if it fails
        print("FFmpeg Error:", e.stderr.decode('utf8'))
        raise

    # 4. Return the sorted list of generated chunk filepaths for the orchestrator
    chunks = sorted(glob.glob(os.path.join(output_dir, "chunk_*.mp4")))
    return chunks