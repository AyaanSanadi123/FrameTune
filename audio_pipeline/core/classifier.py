# audio_pipeline/core/classifier.py

import torch
import torch.nn.functional as F
from languagebind import LanguageBindText, LanguageBindTextTokenizer
from ..config import audio_settings

class ZeroShotAudioAnalyzer:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Analyzer] Loading Text Foundation Model on {self.device}...")
        
        self.tokenizer = LanguageBindTextTokenizer.from_pretrained(audio_settings.AIFM_MODEL_ID)
        self.text_model = LanguageBindText.from_pretrained(
            audio_settings.AIFM_MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.text_model.eval()
        
        # 1. Setup Categories
        self.category_keys = list(audio_settings.ACOUSTIC_CATEGORIES.keys())
        self.category_embeddings = self._embed_and_average(list(audio_settings.ACOUSTIC_CATEGORIES.values()))
        
        # 2. Setup 2D Coordinates
        self.coord_anchors = {
            "energy_high": self._embed_and_average([audio_settings.A_ENERGY_HIGH])[0],
            "energy_low": self._embed_and_average([audio_settings.A_ENERGY_LOW])[0],
            "mood_bright": self._embed_and_average([audio_settings.A_MOOD_BRIGHT])[0],
            "mood_dark": self._embed_and_average([audio_settings.A_MOOD_DARK])[0]
        }

    def _embed_and_average(self, prompt_lists: list[list[str]]) -> torch.Tensor:
        """Embeds arrays of prompts and averages them into stable targets."""
        averaged_anchors = []
        
        for prompts in prompt_lists:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                mean_embed = outputs.text_embeds.mean(dim=0, keepdim=True)
                norm_embed = F.normalize(mean_embed, p=2, dim=-1)
                averaged_anchors.append(norm_embed.squeeze(0))
                
        return torch.stack(averaged_anchors)

    def analyze(self, audio_vector: list[float]) -> dict:
        """Returns the category label and the 2D coordinates."""
        audio_tensor = torch.tensor(audio_vector, dtype=self.text_model.dtype, device=self.device)
        
        # 1. Classify Category
        similarities = audio_tensor @ self.category_embeddings.T
        best_match_idx = similarities.argmax().item()
        category_label = self.category_keys[best_match_idx]
        
        # 2. Project 2D Coordinates via Softmax (0.0 to 1.0 bounds)
        sim_e_low = torch.dot(audio_tensor, self.coord_anchors["energy_low"])
        sim_e_high = torch.dot(audio_tensor, self.coord_anchors["energy_high"])
        energy_val = float(F.softmax(torch.stack([sim_e_low, sim_e_high]), dim=0)[1].item())
        
        sim_m_dark = torch.dot(audio_tensor, self.coord_anchors["mood_dark"])
        sim_m_bright = torch.dot(audio_tensor, self.coord_anchors["mood_bright"])
        mood_val = float(F.softmax(torch.stack([sim_m_dark, sim_m_bright]), dim=0)[1].item())
        
        return {
            "category": category_label,
            "energy": round(energy_val, 4),
            "mood": round(mood_val, 4)
        }