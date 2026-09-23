TARGET_RESOLUTION = (640, 480)  # 480p normalization[cite: 2]
TARGET_FPS = 30                 # Standardized frame rate[cite: 2]
CHUNK_DURATION_SEC = 5          # 5 to 10 second segmentation[cite: 2]
FRAMES_PER_CHUNK = TARGET_FPS * CHUNK_DURATION_SEC



M_MIN = 0.0
M_MAX = 50.0  

C_MIN = 0
C_MAX = 15



CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Custom text anchors acting as directional poles on the Mood axis[cite: 2].
# Instead of one string, we use a list of literal and atmospheric descriptions
T_BRIGHT_PROMPTS = [
    "a bright, uplifting, joyful, and energetic scene",
    "a brightly lit, sunlit environment with vibrant colors",
    "a highly energetic and fast-paced action sequence",
    "a happy, positive, and lighthearted video clip",
    "a cinematic shot with high-key lighting and warm tones"
]

T_DARK_PROMPTS = [
    "a dark, heavy, bleak, and depressing scene",
    "a dimly lit, shadowy environment with muted colors",
    "a slow, tense, and moody cinematic sequence",
    "a somber, melancholic, and serious video clip",
    "a cinematic shot with low-key lighting and cold tones"
]


W_MOTION = 0.7   # Raw pixel displacement influence[cite: 2]
W_CUTS = 0.3     # Editing pace influence[cite: 2]

# Mood Weights: w3 + w4 must = 1.0
W_SEMANTIC = 0.8 
W_BRIGHT = 0.2