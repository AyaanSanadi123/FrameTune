import os
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from ..config.audio_settings import DEMUCS_MODEL, ISOLATED_STEMS

class StemSeparator:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(DEMUCS_MODEL)
        self.model.to(self.device)
        self.model.eval()

    def create_instrumental(self, audio_path: str, output_dir: str) -> str:
        """
        Strips vocals and sums the remaining instruments into a single mix.
        Returns the filepath to the clean instrumental .wav
        """
        os.makedirs(output_dir, exist_ok=True)
        waveform, sr = torchaudio.load(audio_path)

        # Force 44.1kHz stereo for Demucs
        if sr != self.model.samplerate:
            waveform = torchaudio.transforms.Resample(sr, self.model.samplerate)(waveform)
            sr = self.model.samplerate
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
            
        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()
        
        # apply_model outputs tensor shape: (4, channels, time) for [drums, bass, other, vocals]
        sources = apply_model(self.model, waveform[None], device=self.device)[0]
        
        # Initialize an empty canvas for the new mix
        instrumental_mix = torch.zeros_like(sources[0])
        
        # Re-sum everything except the vocals
        for name, source in zip(self.model.sources, sources):
            if name in ISOLATED_STEMS:
                instrumental_mix += source

        # Invert normalization and save
        instrumental_mix = instrumental_mix * ref.std() + ref.mean()
        out_path = os.path.join(output_dir, "instrumental_mix.wav")
        torchaudio.save(out_path, instrumental_mix.cpu(), sr)
        
        return out_path