# core/opencv_engine.py

import cv2
import numpy as np
from config.settings import TARGET_FPS, CUT_DETECTION_THRESHOLD

def process_physical_metrics(filepath: str) -> dict:
    """
    Extracts motion magnitude, cut density, and physical brightness from a video chunk.
    Returns None if the chunk is too short to calculate a physical difference.
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {filepath}")

    motion_scores = []
    brightness_scores = []
    cut_count = 0

    prev_gray = None
    prev_hist = None
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 1. Color Space Conversions
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 2. Physical Brightness (HSV)
            # Normalize 8-bit channels (0-255) to 0.0 - 1.0 and average
            v_channel = hsv[:, :, 2]
            s_channel = hsv[:, :, 1]
            frame_brightness = (np.mean(v_channel) + np.mean(s_channel)) / (2 * 255.0)
            brightness_scores.append(frame_brightness)
            
            # 3. Motion Magnitude (Frame Differencing)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                # Apply a binary threshold at 20 to suppress low-level digital sensor noise
                _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
                # Calculate the exact percentage of pixels that changed in this frame
                motion_score = (np.sum(thresh == 255) / thresh.size) * 100.0
                motion_scores.append(motion_score)
                
            # 4. Cut Density (Histogram Differencing)
            # Calculate a 2D Hue/Saturation histogram and normalize it
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            if prev_hist is not None:
                # cv2.HISTCMP_CORREL scores a 1.0 for a perfect match, and drops for mismatches
                similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                
                # If similarity drops below our heuristic, it registers as a scene cut
                if similarity < CUT_DETECTION_THRESHOLD:
                    cut_count += 1
                    
            # Update trackers for the next frame
            prev_gray = gray
            prev_hist = hist

    finally:
        cap.release()

    # 5. Failsafe: Drop runt chunks that lack enough frames for a difference calculation
    if frame_count < 2:
        return None

    # 6. Mathematical Aggregation
    # Extract the 90th percentile to ignore 1-frame glitches but capture peak action
    final_motion = float(np.percentile(motion_scores, 90)) if motion_scores else 0.0
    final_brightness = float(np.mean(brightness_scores)) if brightness_scores else 0.0

    return {
        "motion_magnitude": final_motion,
        "cut_density": cut_count,
        "physical_brightness": final_brightness
    }