# audio_pipeline/config/audio_settings.py

# ---------------------------------------------------------
# 1. AUDIO DSP & SEGMENTATION
# ---------------------------------------------------------
SAMPLE_RATE = 44100
MIN_LOOP_DURATION = 2.0   # Minimum acceptable stem slice in seconds
MAX_LOOP_DURATION = 8.0   # Maximum acceptable stem slice in seconds

# Demucs stem isolation model
DEMUCS_MODEL = "htdemucs"
ISOLATED_STEMS = ["drums", "bass", "other"]  # "vocals" strictly excluded

# ---------------------------------------------------------
# 2. AUDIO FOUNDATION MODEL (AIFM)
# ---------------------------------------------------------
AIFM_MODEL_ID = "LanguageBind/LanguageBind_Audio"

# ---------------------------------------------------------
# 3. ACOUSTIC CATEGORIES (DEEP TEXT ANCHORS)
# ---------------------------------------------------------
# Text anchors for zero-shot acoustic categorization via LanguageBind
# Expanded to match the ViFM multi-prompt array structure
ACOUSTIC_CATEGORIES = {
    "ambient_bed": [
        "a slow, calm atmospheric ambient synth drone",
        "peaceful background room tone with minimal motion and serene pacing",
        "a static, quiet, cinematic ambient soundscape"
    ],
    "rhythmic_loop": [
        "a fast, punchy, energetic drum loop and percussion groove",
        "frantic, high-speed rhythmic action sequence audio",
        "driving, up-tempo drum patterns with heavy transient energy"
    ],
    "bass_groove": [
        "a heavy sub bass and 808 bassline driving a low frequency groove",
        "deep, rhythmic synth bass with a steady, driving pulse",
        "dark, heavy, and ominous low-key bass instrumentals"
    ],
    "cinematic_sfx": [
        "an intense cinematic riser, impact, and swoosh sound effect",
        "a sudden, high-energy dramatic musical transition",
        "heavy, bleak, tense cinematic sound design and impacts"
    ]
}