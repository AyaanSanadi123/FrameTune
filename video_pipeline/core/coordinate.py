# video_pipeline/core/coordinate.py

import torch
import torch.nn.functional as F
from languagebind import LanguageBindText, LanguageBindTextTokenizer

class UniversalCoordinateSpace:
    """
    A unified 2D mathematical grid based on Russell's Circumplex Model.
    Projects raw high-dimensional embeddings down to a [0.0 - 1.0] plane.
    """
    def __init__(self, device: str, anchors_config):
        self.device = device
        print(f"[Coordinate Space] Loading Text Foundation Model on {self.device}...")
        
        # 1. Load the LanguageBind Text Model (Matching ViFM precision)
        self.tokenizer = LanguageBindTextTokenizer.from_pretrained(anchors_config.VIFM_MODEL_ID)
        self.text_model = LanguageBindText.from_pretrained(
            anchors_config.VIFM_MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.text_model.eval()
        
        # 2. Pre-calculate the absolute poles of the 2D coordinate plane
        self.anchors = {
            "energy_high": self._embed_text(anchors_config.T_ENERGY_HIGH),
            "energy_low": self._embed_text(anchors_config.T_ENERGY_LOW),
            "mood_bright": self._embed_text(anchors_config.T_MOOD_BRIGHT),
            "mood_dark": self._embed_text(anchors_config.T_MOOD_DARK)
        }

    def _embed_text(self, prompts: list) -> torch.Tensor:
        """Helper to tokenize, embed, average, and normalize text anchors."""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            # FIXED: LanguageBind syntax
            outputs = self.text_model(**inputs)
            features = outputs.text_embeds
            
            mean_feature = features.mean(dim=0, keepdim=True)
            return F.normalize(mean_feature, p=2, dim=-1)

    def project_embedding(self, raw_embedding: torch.Tensor) -> dict:
        """Takes a raw spatiotemporal vector and projects it onto the axes."""
        # Ensure identical tensor precision before dot product
        raw_embedding = raw_embedding.to(dtype=self.text_model.dtype, device=self.device)
        norm_embedding = F.normalize(raw_embedding, p=2, dim=-1)
        
        # Axis 1: Energy Projection
        sim_e_low = torch.matmul(norm_embedding, self.anchors["energy_low"].T).squeeze()
        sim_e_high = torch.matmul(norm_embedding, self.anchors["energy_high"].T).squeeze()
        energy_probs = F.softmax(torch.stack([sim_e_low, sim_e_high]), dim=0)
        
        # Axis 2: Mood Projection
        sim_m_dark = torch.matmul(norm_embedding, self.anchors["mood_dark"].T).squeeze()
        sim_m_bright = torch.matmul(norm_embedding, self.anchors["mood_bright"].T).squeeze()
        mood_probs = F.softmax(torch.stack([sim_m_dark, sim_m_bright]), dim=0)
        
        return {
            "energy": float(energy_probs[1].item()),
            "mood": float(mood_probs[1].item())
        }

    def build_trajectory(self, temporal_metadata: list, raw_embeddings: list) -> list:
        trajectory = []
        for meta, embedding in zip(temporal_metadata, raw_embeddings):
            coords = self.project_embedding(embedding)
            trajectory.append({
                "chunk_order": meta.get("chunk_order", 0),
                "start_time": meta["start_time"],
                "duration": meta["duration"],
                "ui_coordinate": [coords["mood"], coords["energy"]],
                "embedding": embedding.cpu().numpy().tolist()
            })
        return trajectory