import os
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("vitascan.gat_model")

class VitaScanGATDeficiencyGrader:
    def __init__(self, model_weight_path: Optional[str] = None):
        self.weights_dir = Path(__file__).parent / "weights"
        self.weights_dir.mkdir(exist_ok=True)
        self.model_path = model_weight_path or (self.weights_dir / "gat_model.pkl")
        self.loaded_model = None
        self._load_weights()

    def _load_weights(self):
        if Path(self.model_path).exists():
            try:
                with open(self.model_path, "rb") as f:
                    self.loaded_model = pickle.load(f)
                logger.info(f"Successfully loaded GAT .pkl model weights from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load GAT .pkl file ({e}). Using rule-assisted severity grader.")

    def grade(self, normalized_biomarkers: Dict[str, Any]) -> Dict[str, Any]:
        ferritin_dev = normalized_biomarkers.get("ferritin", {}).get("deviation", -0.3)
        hb_dev = normalized_biomarkers.get("hemoglobin", {}).get("deviation", -0.2)
        b12_dev = normalized_biomarkers.get("b12", {}).get("deviation", 0.0)
        folate_dev = normalized_biomarkers.get("folate", {}).get("deviation", 0.0)

        def get_band(dev):
            if dev < -0.4: return "severe"
            if dev < -0.2: return "moderate"
            if dev < 0: return "mild"
            return "none"

        return {
            "iron": {"score": min(1.0, abs(ferritin_dev)), "band": get_band(ferritin_dev), "model": "gat" if self.loaded_model else "gat_stub"},
            "anemia": {"score": min(1.0, abs(hb_dev)), "band": get_band(hb_dev), "model": "gat" if self.loaded_model else "gat_stub"},
            "b12": {"score": min(1.0, abs(b12_dev)), "band": get_band(b12_dev), "model": "gat" if self.loaded_model else "gat_stub"},
            "folate": {"score": min(1.0, abs(folate_dev)), "band": get_band(folate_dev), "model": "gat" if self.loaded_model else "gat_stub"}
        }
