# audio_pipeline/config/audio_settings.py

SAMPLE_RATE = 44100
MIN_LOOP_DURATION = 2.0   # Minimum acceptable stem slice in seconds
MAX_LOOP_DURATION = 8.0   # Maximum acceptable stem slice in seconds

# Text anchors for zero-shot acoustic categorization via LanguageBind
ACOUSTIC_CATEGORIES = {
    "ambient_bed": "calm atmospheric ambient synth drone background room tone",
    "rhythmic_loop": "fast punchy energetic drum loop percussion groove",
    "bass_groove": "heavy sub bass 808 bassline driving low frequency synth",
    "cinematic_sfx": "intense cinematic riser impact swoosh sound effect transition"
}

# Demucs stem isolation model
DEMUCS_MODEL = "htdemucs"
ISOLATED_STEMS = ["drums", "bass", "other"]  # "vocals" strictly excluded