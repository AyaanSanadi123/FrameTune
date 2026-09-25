# core/coordinate_map.py

import torch
import torch.nn.functional as F

class UniversalCoordinateSpace:
    """
    A unified 2D mathematical grid based on Russell's Circumplex Model.
    Projects raw high-dimensional embeddings (from Video or Audio) down to a [0.0 - 1.0] plane.
    """
    def __init__(self, model, processor, device, anchors_config):
        self.model = model
        self.processor = processor
        self.device = device
        
        # Pre-calculate the absolute poles of the 2D coordinate plane
        self.anchors = {
            "energy_high": self._embed_text(anchors_config.T_ENERGY_HIGH),
            "energy_low": self._embed_text(anchors_config.T_ENERGY_LOW),
            "mood_bright": self._embed_text(anchors_config.T_MOOD_BRIGHT),
            "mood_dark": self._embed_text(anchors_config.T_MOOD_DARK)
        }

    def _embed_text(self, prompts: list) -> torch.Tensor:
        """Helper to tokenize, embed, average, and normalize text anchors."""
        inputs = self.processor(
            text=prompts,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            # Average the sentences into a single vector, then normalize
            mean_feature = features.mean(dim=0, keepdim=True)
            return F.normalize(mean_feature, p=2, dim=-1)

    def project_embedding(self, raw_embedding: torch.Tensor) -> dict:
        """
        Takes a raw 512D spatiotemporal vector and projects it onto the Energy/Mood axes.
        Returns a dictionary with the final 2D coordinates.
        """
        # Ensure the incoming vector is normalized before dot product
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
        """
        Constructs the final JSON payload for the frontend UI.
        Matches the dynamic shot metadata (timestamps) with their computed coordinates.
        """
        trajectory = []
        for meta, embedding in zip(temporal_metadata, raw_embeddings):
            coords = self.project_embedding(embedding)
            trajectory.append({
                "timestamp": meta["start_time"],
                "duration": meta["duration"],
                "coordinate": [coords["mood"], coords["energy"]]
            })
        return trajectory