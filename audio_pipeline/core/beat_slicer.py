import os
import librosa
import soundfile as sf
import numpy as np
from ..config.audio_settings import MIN_LOOP_DURATION, MAX_LOOP_DURATION

def extract_rhythmic_loops(audio_path: str, track_id: str, output_dir: str) -> list[dict]:
    """
    Analyzes transients and beats of an instrumental track, 
    preserving full stereo fidelity, and slices it into bar-aligned loops.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load full-fidelity audio (preserve native sample rate and stereo channels)
    # y shape: (channels, samples) if stereo, or (samples,) if mono
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    
    # Create a temporary mono downmix strictly for beat/transient tracking
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y

    # 2. Beat and tempo estimation on the mono envelope
    tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr, trim=False)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    bpm = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)

    # Fallback to uniform windowing if the track lacks strong transient onsets
    if len(beat_times) < 8:
        return _slice_ambient_intervals(y, sr, track_id, output_dir, bpm)

    # 3. Slice audio into 4-bar musical phrases (16 beats in 4/4 time)
    beats_per_segment = 16
    slices = []

    for i in range(0, len(beat_times) - beats_per_segment, beats_per_segment):
        t_start = beat_times[i]
        t_end = beat_times[i + beats_per_segment]
        duration = t_end - t_start

        if MIN_LOOP_DURATION <= duration <= MAX_LOOP_DURATION:
            idx_start = int(t_start * sr)
            idx_end = int(t_end * sr)

            # Slice stereo channels: (channels, slice_samples)
            segment_data = y[:, idx_start:idx_end] if y.ndim > 1 else y[idx_start:idx_end]

            # 10ms micro-fade to eliminate zero-crossing pops
            segment_data = _apply_micro_fades(segment_data, sr, fade_duration=0.010)

            loop_filename = f"{track_id}_loop_{i:04d}.wav"
            loop_path = os.path.join(output_dir, loop_filename)

            # soundfile expects (samples, channels) for stereo writes
            out_waveform = segment_data.T if segment_data.ndim > 1 else segment_data
            sf.write(loop_path, out_waveform, sr)

            slices.append({
                "filepath": loop_path,
                "duration": round(duration, 3),
                "bpm": round(bpm, 2),
                "track_id": track_id
            })

    return slices

def _slice_ambient_intervals(y: np.ndarray, sr: int, track_id: str, output_dir: str, bpm: float) -> list[dict]:
    """Uniform windowing for ambient passages lacking distinct rhythmic beats."""
    step_duration = 6.0
    step_samples = int(step_duration * sr)
    total_samples = y.shape[-1]
    slices = []

    for i, start_sample in enumerate(range(0, total_samples - step_samples, step_samples)):
        end_sample = start_sample + step_samples
        segment_data = y[:, start_sample:end_sample] if y.ndim > 1 else y[start_sample:end_sample]

        # 25ms micro-fade for softer ambient transitions
        segment_data = _apply_micro_fades(segment_data, sr, fade_duration=0.025)

        loop_filename = f"{track_id}_ambient_{i:04d}.wav"
        loop_path = os.path.join(output_dir, loop_filename)

        out_waveform = segment_data.T if segment_data.ndim > 1 else segment_data
        sf.write(loop_path, out_waveform, sr)

        slices.append({
            "filepath": loop_path,
            "duration": round(step_duration, 3),
            "bpm": round(bpm, 2) if bpm > 0 else 0.0,
            "track_id": track_id
        })

    return slices

def _apply_micro_fades(audio: np.ndarray, sr: int, fade_duration: float = 0.010) -> np.ndarray:
    """Applies smooth linear micro-fades supporting both 1D (mono) and 2D (stereo) arrays."""
    fade_len = int(fade_duration * sr)
    num_samples = audio.shape[-1]

    if num_samples <= 2 * fade_len:
        return audio

    fade_in = np.linspace(0.0, 1.0, fade_len)
    fade_out = np.linspace(1.0, 0.0, fade_len)

    if audio.ndim == 1:
        audio[:fade_len] *= fade_in
        audio[-fade_len:] *= fade_out
    else:
        # Broadcast fade curve across all audio channels
        audio[:, :fade_len] *= fade_in
        audio[:, -fade_len:] *= fade_out

    return audio