"""Hetero-GNN reconstruction — exact replica of ``HeteroClinicalGNN``.

Spec taken from docs/model.py (training script) and validated against the
shapes in ``hetero_gnn_weights.pt``:
  2 × HeteroConv(SAGEConv, aggr='mean') over relations
    (patient, measures, biomarker), (biomarker, rev_measures, patient),
    (biomarker, indicates, etiology), (etiology, rev_indicates, biomarker)
  + classifier MLP (64 -> ReLU -> Dropout0.2 -> 4).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric import transforms as T

from . import features as FEA

_knowledge_edges: list[tuple[int, int]] = []


class HeteroClinicalGNN(nn.Module):
    def __init__(self, num_features, num_classes=4, hidden_dim=64, num_biomarkers=12):
        super().__init__()
        self.patient_proj = Linear(num_features, hidden_dim)
        self.biomarker_proj = Linear(num_biomarkers, hidden_dim)
        self.etiology_proj = Linear(num_classes, hidden_dim)

        relations = {
            ("patient", "measures", "biomarker"): SAGEConv((-1, -1), hidden_dim),
            ("biomarker", "rev_measures", "patient"): SAGEConv((-1, -1), hidden_dim),
            ("biomarker", "indicates", "etiology"): SAGEConv((-1, -1), hidden_dim),
            ("etiology", "rev_indicates", "biomarker"): SAGEConv((-1, -1), hidden_dim),
        }
        self.conv1 = HeteroConv(relations, "mean")
        self.conv2 = HeteroConv(relations, "mean")

        self.classifier = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict["patient"] = F.relu(self.patient_proj(x_dict["patient"]))
        x_dict["biomarker"] = F.relu(self.biomarker_proj(x_dict["biomarker"]))
        x_dict["etiology"] = F.relu(self.etiology_proj(x_dict["etiology"]))

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.elu(x) for k, x in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.elu(x) for k, x in x_dict.items()}

        return self.classifier(x_dict["patient"])


def build_hetero_clinical_graph(df_scaled) -> HeteroData:
    """Replicates ``build_hetero_clinical_graph`` from the training script.

    ``df_scaled`` has exactly the 12 FEATURE_NAMES columns (already scaled).
    Multiple patients share the 12 biomarker nodes and 4 etiology nodes, exactly
    as in training-time construction.
    """
    num_biomarkers = len(FEA.FEATURE_NAMES)
    num_etiologies = 4

    data = HeteroData()
    data["patient"].x = torch.tensor(
        df_scaled[FEA.FEATURE_NAMES].to_numpy(dtype=float), dtype=torch.float
    )
    data["biomarker"].x = torch.eye(num_biomarkers, dtype=torch.float)
    data["etiology"].x = torch.eye(num_etiologies, dtype=torch.float)

    num_patients = len(df_scaled)
    p_idx, b_idx = [], []
    for p in range(num_patients):
        for b in range(num_biomarkers):
            p_idx.append(p)
            b_idx.append(b)
    data["patient", "measures", "biomarker"].edge_index = torch.stack(
        [torch.tensor(p_idx, dtype=torch.long), torch.tensor(b_idx, dtype=torch.long)], dim=0
    )

    edges = _knowledge_edges or []
    if edges:
        k_src = [e[0] for e in edges]
        k_dst = [e[1] for e in edges]
        data["biomarker", "indicates", "etiology"].edge_index = torch.tensor(
            [k_src, k_dst], dtype=torch.long
        )

    return T.ToUndirected()(data)


def load_gnn(model_dir):
    """Load metadata + weights, reconstruct the model, return (model, metadata)."""
    md_path = model_dir / "gnn_metadata.pt"
    wts_path = model_dir / "hetero_gnn_weights.pt"

    metadata = torch.load(md_path, map_location="cpu", weights_only=True)
    global _knowledge_edges
    _knowledge_edges = [tuple(int(a) for a in e) for e in metadata["knowledge_edges"]]

    model = HeteroClinicalGNN(
        num_features=int(metadata["num_features"]),
        num_classes=int(metadata["num_classes"]),
        hidden_dim=int(metadata["hidden_dim"]),
    )
    state = torch.load(wts_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, metadata