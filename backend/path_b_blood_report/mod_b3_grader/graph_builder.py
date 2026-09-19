from typing import Dict, Any

class BiomarkerGraphBuilder:
    def build_graph(self, normalized_biomarkers: Dict[str, Any]):
        nodes = list(normalized_biomarkers.keys())
        features = [[b_info["raw_value"], b_info["deviation"], 1.0 if b_info["is_abnormal"] else 0.0] for b_info in normalized_biomarkers.values()]
        return {"nodes": nodes, "features": features}
