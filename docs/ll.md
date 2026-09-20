I have trained a hybrid Heterogeneous Graph Neural Network (Hetero-GNN) + Hierarchical XGBoost Cascade model for multi-class anemia diagnosis. I need you to write a standalone test script `test_inference.py` to verify that my saved model artifacts can be loaded successfully and generate correct predictions on new patient sample data.

### 1. Saved Artifacts Available in Working Directory
- `knn_imputer.joblib` (KNN Imputer for raw features)
- `standard_scaler.joblib` (StandardScaler fitted on imputed features)
- `xgb_cascade.joblib` (Fitted HierarchicalXGBoostCascade instance)
- `hetero_gnn_weights.pt` (PyTorch state_dict for HeteroClinicalGNN)
- `gnn_metadata.pt` (Saved metadata or PyG graph transformation parameters)
- `pipeline_config.joblib` (Saved configuration dictionary with thresholds, alpha, etc.)

### 2. Feature Schema & Biomarkers
- **Base Features (9)**: `hemoglobin`, `mcv`, `rdw`, `ferritin`, `b12`, `folate`, `rbc`, `mch`, `mchc`
- **Calculated Biomarkers (3)**:
  - `srivastava_index` = `mch` / (`rbc` + 1e-6)
  - `mentzer_index` = `mcv` / (`rbc` + 1e-6)
  - `cell_hb_density` = (`mch` * `mchc`) / 100.0
- **Target Etiologies**: `0: No Anemia`, `1: IDA`, `2: B12 Deficiency`, `3: Folate Deficiency`, `4: Grey Zone Triage`

### 3. Model & Graph Architecture Definitions
Please include the following class definitions in `test_inference.py` so PyTorch and Joblib can deserialize the models:

```python
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, Linear, SAGEConv
from xgboost import XGBClassifier

BIOMARKER_NAMES = [
    'hemoglobin',
    'mcv',
    'rdw',
    'ferritin',
    'b12',
    'folate',
    'rbc',
    'mch',
    'mchc',
    'srivastava_index',
    'mentzer_index',
    'cell_hb_density',
]
ETIOLOGY_NAMES = [
    'No Anemia',
    'IDA',
    'B12 Deficiency',
    'Folate Deficiency',
    'Grey Zone Triage',
]

CLINICAL_KNOWLEDGE_EDGES = [
    (BIOMARKER_NAMES.index('ferritin'), 1),
    (BIOMARKER_NAMES.index('srivastava_index'), 1),
    (BIOMARKER_NAMES.index('mentzer_index'), 1),
    (BIOMARKER_NAMES.index('b12'), 2),
    (BIOMARKER_NAMES.index('folate'), 3),
    (BIOMARKER_NAMES.index('mcv'), 2),
    (BIOMARKER_NAMES.index('mcv'), 3),
]


class HierarchicalXGBoostCascade:

  def __init__(self):
    pass


class HeteroClinicalGNN(nn.Module):

  def __init__(self, num_features, num_classes=4, hidden_dim=64):
    super(HeteroClinicalGNN, self).__init__()
    self.patient_proj = Linear(num_features, hidden_dim)
    self.biomarker_proj = Linear(len(BIOMARKER_NAMES), hidden_dim)
    self.etiology_proj = Linear(num_classes, hidden_dim)

    self.conv1 = HeteroConv(
        {
            ('patient', 'measures', 'biomarker'): SAGEConv(
                (-1, -1), hidden_dim
            ),
            ('biomarker', 'rev_measures', 'patient'): SAGEConv(
                (-1, -1), hidden_dim
            ),
            ('biomarker', 'indicates', 'etiology'): SAGEConv(
                (-1, -1), hidden_dim
            ),
            ('etiology', 'rev_indicates', 'biomarker'): SAGEConv(
                (-1, -1), hidden_dim
            ),
        },
        'mean',
    )

    self.conv2 = HeteroConv(
        {
            ('patient', 'measures', 'biomarker'): SAGEConv(
                (-1, -1), hidden_dim
            ),
            ('biomarker', 'rev_measures', 'patient'): SAGEConv(
                (-1, -1), hidden_dim
            ),
            ('biomarker', 'indicates', 'etiology'): SAGEConv(
                (-1, -1), hidden_dim
            ),
            ('etiology', 'rev_indicates', 'biomarker'): SAGEConv(
                (-1, -1), hidden_dim
            ),
        },
        'mean',
    )

    self.classifier = nn.Sequential(
        Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        Linear(hidden_dim // 2, num_classes),
    )

  def forward(self, x_dict, edge_index_dict):
    x_dict['patient'] = F.relu(self.patient_proj(x_dict['patient']))
    x_dict['biomarker'] = F.relu(self.biomarker_proj(x_dict['biomarker']))
    x_dict['etiology'] = F.relu(self.etiology_proj(x_dict['etiology']))

    x_dict = self.conv1(x_dict, edge_index_dict)
    x_dict = {key: F.elu(x) for key, x in x_dict.items()}

    x_dict = self.conv2(x_dict, edge_index_dict)
    x_dict = {key: F.elu(x) for key, x in x_dict.items()}

    return self.classifier(x_dict['patient'])


def build_hetero_clinical_graph(df_scaled):
  data = HeteroData()
  num_patients = len(df_scaled)
  num_biomarkers = len(BIOMARKER_NAMES)
  num_etiologies = 4

  data['patient'].x = torch.tensor(df_scaled.values, dtype=torch.float)
  data['biomarker'].x = torch.eye(num_biomarkers, dtype=torch.float)
  data['etiology'].x = torch.eye(num_etiologies, dtype=torch.float)

  p_idx, b_idx, edge_w = [], [], []
  vals = df_scaled[BIOMARKER_NAMES].values
  for p in range(num_patients):
    for b in range(num_biomarkers):
      p_idx.append(p)
      b_idx.append(b)
      edge_w.append(vals[p, b])

  data['patient', 'measures', 'biomarker'].edge_index = torch.stack(
      [torch.tensor(p_idx, dtype=torch.long), torch.tensor(b_idx, dtype=torch.long)],
      dim=0,
  )
  data['patient', 'measures', 'biomarker'].edge_attr = torch.tensor(
      edge_w, dtype=torch.float
  ).unsqueeze(1)

  k_src = [e[0] for e in CLINICAL_KNOWLEDGE_EDGES]
  k_dst = [e[1] for e in CLINICAL_KNOWLEDGE_EDGES]
  data['biomarker', 'indicates', 'etiology'].edge_index = torch.tensor(
      [k_src, k_dst], dtype=torch.long
  )

  return T.ToUndirected()(data)

4. What test_inference.py Should Do

    Load Artifacts: Load knn_imputer.joblib, standard_scaler.joblib, xgb_cascade.joblib, initialize HeteroClinicalGNN, and load weights from hetero_gnn_weights.pt.

    Create 5 Synthetic Test Patients:

        Patient 1 (Healthy): Hb=14.0, MCV=88, Ferritin=90, B12=450, Folate=12.0, RBC=4.5, MCH=31, MCHC=34, RDW=12.0

        Patient 2 (Severe IDA): Hb=8.5, MCV=70, Ferritin=6, B12=350, Folate=10.0, RBC=3.8, MCH=22, MCHC=28, RDW=16.5

        Patient 3 (B12 Deficiency): Hb=9.5, MCV=105, Ferritin=100, B12=110, Folate=11.0, RBC=3.1, MCH=34, MCHC=33, RDW=15.5

        Patient 4 (Folate Deficiency): Hb=9.8, MCV=102, Ferritin=85, B12=380, Folate=2.1, RBC=3.2, MCH=33, MCHC=33, RDW=15.0

        Patient 5 (Borderline/Grey Zone): Hb=10.5, MCV=85, Ferritin=np.nan, B12=220, Folate=8.0, RBC=3.6, MCH=29, MCHC=32, RDW=14.5

    Execute Preprocessing:

        Calculate missing calculated indices (srivastava_index, mentzer_index, cell_hb_density).

        Run KNN Imputer and Standard Scaler.

    Execute Inference:

        Predict probabilities from XGBoost Cascade.

        Build PyG HeteroGraph and get softmax probabilities from HeteroClinicalGNN.

        Blend probabilities (alpha = 0.50).

        Run Decision Logic (P_Healthy >= 0.50 -> Healthy; P_Folate >= 0.40 -> Folate; conditional B12 check with bounds 0.35 to 0.65 for Grey Zone).

    Print Outputs: Display a clean tabular summary showing Patient ID, Input Characteristics, XGBoost Probs, GNN Probs, Blended Probs, and Final Predicted Diagnosis.

Write the clean, modular Python code for test_inference.py now.