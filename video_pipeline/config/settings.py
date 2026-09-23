# config/settings.py

"""
FrameTune Configuration (ViFM Architecture)
Centralizes heuristics, models, and 2D text anchors for the spatiotemporal pipeline.
"""

# ---------------------------------------------------------
# 1. VIDEO INGESTION & SEGMENTATION
# ---------------------------------------------------------
TARGET_RESOLUTION = (640, 480)  
TARGET_FPS = 30                 
TRANSNET_THRESHOLD = 0.5       # Sensitivity for cinematic cut detection [1]

# ---------------------------------------------------------
# 2. VIDEO FOUNDATION MODEL (ViFM)
# ---------------------------------------------------------
# 3D Video-Language Model for spatiotemporal feature extraction [2]
VIFM_MODEL_ID = "LanguageBind/LanguageBind_Video_merge"
FRAMES_PER_CLIP = 8            # Number of frames the 3D model extracts per chunk to build the "tubelet"

# ---------------------------------------------------------
# 3. COORDINATE ANCHORS (ENERGY & MOOD)
# ---------------------------------------------------------
# The model maps the video chunk directly against these two axes

# Y-Axis (Energy / Arousal)
T_ENERGY_HIGH = [
    "a frantic, chaotic, high-speed action sequence with rapid movement",
    "intense, fast-paced cinematic motion, shaky camera, and high energy"
]

T_ENERGY_LOW = [
    "a slow, static, peaceful, calm, and perfectly still scene",
    "minimal motion, steady camera, serene pacing, and low energy"
]

# X-Axis (Mood / Valence)
T_MOOD_BRIGHT = [
    "a bright, joyful, warm, uplifting scene with positive emotion",
    "high-key lighting, vibrant colors, and a happy, lighthearted atmosphere"
]

T_MOOD_DARK = [
    "a dark, heavy, bleak, somber, ominous scene with negative emotion",
    "low-key lighting, muted shadows, and a tense, depressive atmosphere"
]