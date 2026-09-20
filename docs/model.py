import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

warnings.filterwarnings('ignore')

SEED = 467
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. BALANCED SYNTHETIC DATA GENERATOR
# ==============================================================================
BIOMARKER_NAMES = [
    'hemoglobin', 'mcv', 'rdw', 'ferritin', 'b12', 'folate',
    'rbc', 'mch', 'mchc', 'srivastava_index', 'mentzer_index', 'cell_hb_density'
]
ETIOLOGY_NAMES = ['No Anemia', 'IDA', 'B12 Deficiency', 'Folate Deficiency']

def generate_balanced_clinical_cohort(n_samples=3000):
    np.random.seed(SEED)
    y = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.40, 0.30, 0.15, 0.15])
   
    hb, mcv, rdw = np.zeros(n_samples), np.zeros(n_samples), np.zeros(n_samples)
    ferritin, b12, folate = np.zeros(n_samples), np.zeros(n_samples), np.zeros(n_samples)
    rbc, mch, mchc = np.zeros(n_samples), np.zeros(n_samples), np.zeros(n_samples)

    for i in range(n_samples):
        if y[i] == 0:  # No Anemia
            hb[i] = np.random.normal(13.5, 1.0)
            mcv[i] = np.random.normal(88.0, 4.0)
            ferritin[i] = np.random.normal(80.0, 20.0)
            b12[i] = np.random.normal(400.0, 80.0)
            folate[i] = np.random.normal(12.0, 2.0)
        elif y[i] == 1:  # IDA
            hb[i] = np.random.normal(9.5, 1.2)
            mcv[i] = np.random.normal(74.0, 5.0)
            ferritin[i] = np.random.exponential(10.0)
            b12[i] = np.random.normal(350.0, 70.0)
            folate[i] = np.random.normal(10.0, 2.0)
        elif y[i] == 2:  # B12 Deficiency
            hb[i] = np.random.normal(10.0, 1.2)
            mcv[i] = np.random.normal(102.0, 6.0)
            ferritin[i] = np.random.normal(90.0, 25.0)
            b12[i] = np.random.normal(120.0, 35.0)
            folate[i] = np.random.normal(10.0, 2.0)
        elif y[i] == 3:  # Folate Deficiency
            hb[i] = np.random.normal(10.2, 1.1)
            mcv[i] = np.random.normal(100.0, 5.0)
            ferritin[i] = np.random.normal(85.0, 20.0)
            b12[i] = np.random.normal(380.0, 60.0)
            folate[i] = np.random.normal(2.5, 0.8)

        rbc[i] = hb[i] / 3.0 + np.random.normal(0, 0.2)
        mch[i] = (hb[i] / (rbc[i] + 1e-6)) * 10
        mchc[i] = np.random.normal(33.0, 1.5)
        rdw[i] = np.random.normal(15.0 if y[i] > 0 else 12.5, 1.5)

    df = pd.DataFrame({
        'hemoglobin': hb, 'mcv': mcv, 'rdw': rdw, 'ferritin': ferritin,
        'b12': b12, 'folate': folate, 'rbc': rbc, 'mch': mch, 'mchc': mchc
    })
   
    eps = 1e-6
    df['srivastava_index'] = df['mch'] / (df['rbc'] + eps)
    df['mentzer_index'] = df['mcv'] / (df['rbc'] + eps)
    df['cell_hb_density'] = (df['mch'] * df['mchc']) / 100.0
   
    # Introduce ~2% random missingness for KNN Imputation simulation
    mask = np.random.rand(*df.shape) < 0.02
    df[mask] = np.nan

    return df, y

# ==============================================================================
# 2. STAGE 1, 2a, 2b HIERARCHICAL XGBOOST CASCADE
# ==============================================================================
class HierarchicalXGBoostCascade:
    def __init__(self):
        self.stage1_xgb = XGBClassifier(n_estimators=100, max_depth=4, eval_metric='logloss', random_state=SEED)
        self.stage2a_xgb = XGBClassifier(n_estimators=100, max_depth=4, scale_pos_weight=3.0, eval_metric='logloss', random_state=SEED)
       
        base_2b = XGBClassifier(n_estimators=100, max_depth=4, eval_metric='logloss', random_state=SEED)
        self.stage2b_calibrated = CalibratedClassifierCV(estimator=base_2b, method='isotonic', cv=3)

    def fit(self, X_train, y_train):
        # Stage 1: Binary Screening (0: Healthy vs 1: Anemia [1,2,3])
        y_stage1 = (y_train > 0).astype(int)
        self.stage1_xgb.fit(X_train, y_stage1)

        # Stage 2a: Folate Isolator (0: Non-Folate Anemia [1,2] vs 1: Folate [3])
        anemic_mask = (y_train > 0)
        X_anemic = X_train[anemic_mask]
        y_anemic = y_train[anemic_mask]
       
        y_stage2a = (y_anemic == 3).astype(int)
        self.stage2a_xgb.fit(X_anemic, y_stage2a)

        # Stage 2b: Calibrated Differential Classifier (0: IDA [1] vs 1: B12 [2])
        diff_mask = (y_train == 1) | (y_train == 2)
        X_diff = X_train[diff_mask]
        y_diff = (y_train[diff_mask] == 2).astype(int) # 0: IDA, 1: B12
        self.stage2b_calibrated.fit(X_diff, y_diff)

    def predict_proba(self, X):
        n_samples = len(X)
        probs_xgb = np.zeros((n_samples, 4))

        p_anemia = self.stage1_xgb.predict_proba(X)[:, 1]
        p_healthy = 1.0 - p_anemia

        p_folate_given_anemia = self.stage2a_xgb.predict_proba(X)[:, 1]
        p_diff_given_anemia = 1.0 - p_folate_given_anemia

        p_b12_given_diff = self.stage2b_calibrated.predict_proba(X)[:, 1]
        p_ida_given_diff = 1.0 - p_b12_given_diff

        # Assemble full 4-class joint probability distribution
        probs_xgb[:, 0] = p_healthy
        probs_xgb[:, 1] = p_anemia * p_diff_given_anemia * p_ida_given_diff
        probs_xgb[:, 2] = p_anemia * p_diff_given_anemia * p_b12_given_diff
        probs_xgb[:, 3] = p_anemia * p_folate_given_anemia

        return probs_xgb

# ==============================================================================
# 3. PYG HETEROGENEOUS KNOWLEDGE GRAPH GNN (BIOMARKER GAT/SAGE)
# ==============================================================================
CLINICAL_KNOWLEDGE_EDGES = [
    (BIOMARKER_NAMES.index('ferritin'), 1),
    (BIOMARKER_NAMES.index('srivastava_index'), 1),
    (BIOMARKER_NAMES.index('mentzer_index'), 1),
    (BIOMARKER_NAMES.index('b12'), 2),
    (BIOMARKER_NAMES.index('folate'), 3),
    (BIOMARKER_NAMES.index('mcv'), 2),
    (BIOMARKER_NAMES.index('mcv'), 3),
]

def build_hetero_clinical_graph(df_scaled, y_labels=None):
    data = HeteroData()
    num_patients = len(df_scaled)
    num_biomarkers = len(BIOMARKER_NAMES)
    num_etiologies = len(ETIOLOGY_NAMES)

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

    data['patient', 'measures', 'biomarker'].edge_index = torch.stack([
        torch.tensor(p_idx, dtype=torch.long),
        torch.tensor(b_idx, dtype=torch.long)
    ], dim=0)
    data['patient', 'measures', 'biomarker'].edge_attr = torch.tensor(edge_w, dtype=torch.float).unsqueeze(1)

    k_src = [e[0] for e in CLINICAL_KNOWLEDGE_EDGES]
    k_dst = [e[1] for e in CLINICAL_KNOWLEDGE_EDGES]
    data['biomarker', 'indicates', 'etiology'].edge_index = torch.tensor([k_src, k_dst], dtype=torch.long)

    if y_labels is not None:
        data['patient'].y = torch.tensor(y_labels, dtype=torch.long)

    return T.ToUndirected()(data)

class HeteroClinicalGNN(nn.Module):
    def __init__(self, num_features, num_classes=4, hidden_dim=64):
        super(HeteroClinicalGNN, self).__init__()
        self.patient_proj = Linear(num_features, hidden_dim)
        self.biomarker_proj = Linear(len(BIOMARKER_NAMES), hidden_dim)
        self.etiology_proj = Linear(num_classes, hidden_dim)

        self.conv1 = HeteroConv({
            ('patient', 'measures', 'biomarker'): SAGEConv((-1, -1), hidden_dim),
            ('biomarker', 'rev_measures', 'patient'): SAGEConv((-1, -1), hidden_dim),
            ('biomarker', 'indicates', 'etiology'): SAGEConv((-1, -1), hidden_dim),
            ('etiology', 'rev_indicates', 'biomarker'): SAGEConv((-1, -1), hidden_dim),
        }, 'mean')

        self.conv2 = HeteroConv({
            ('patient', 'measures', 'biomarker'): SAGEConv((-1, -1), hidden_dim),
            ('biomarker', 'rev_measures', 'patient'): SAGEConv((-1, -1), hidden_dim),
            ('biomarker', 'indicates', 'etiology'): SAGEConv((-1, -1), hidden_dim),
            ('etiology', 'rev_indicates', 'biomarker'): SAGEConv((-1, -1), hidden_dim),
        }, 'mean')

        self.classifier = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(hidden_dim // 2, num_classes)
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()

# ==============================================================================
# 4. FULL PIPELINE EXECUTION & GREY-ZONE TRIAGE ROUTER
# ==============================================================================
def run_full_hybrid_pipeline(alpha=0.50, lower_bound=0.35, upper_bound=0.65):
    print("--- 1. Generating Cohort & Preprocessing ---")
    df_raw, y_raw = generate_balanced_clinical_cohort(n_samples=3000)
   
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df_raw, y_raw, test_size=0.20, random_state=SEED, stratify=y_raw
    )

    # Imputation Path
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns)

    # Scaling Path
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train_imp.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test_imp.columns)

    print("--- 2. Training Stage Cascade (XGBoost) ---")
    cascade = HierarchicalXGBoostCascade()
    cascade.fit(X_train_imp, y_train)
    probs_xgb = cascade.predict_proba(X_test_imp)

    print("--- 3. Constructing & Training HeteroData GNN ---")
    train_graph = build_hetero_clinical_graph(X_train_scaled, y_train).to(device)
    test_graph = build_hetero_clinical_graph(X_test_scaled, y_test).to(device)

    class_counts = np.bincount(y_train, minlength=4)
    class_weights = torch.tensor(len(y_train) / (4.0 * np.maximum(class_counts, 1)), dtype=torch.float).to(device)

    gnn_model = HeteroClinicalGNN(num_features=X_train_scaled.shape[1], num_classes=4, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(gnn_model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    gnn_model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        out = gnn_model(train_graph.x_dict, train_graph.edge_index_dict)
        loss = criterion(out, train_graph['patient'].y)
        loss.backward()
        optimizer.step()

    gnn_model.eval()
    with torch.no_grad():
        test_out = gnn_model(test_graph.x_dict, test_graph.edge_index_dict)
        probs_gat = F.softmax(test_out, dim=1).cpu().numpy()

    print("--- 4. Executing Soft-Voting & Clinical Grey Zone Triage ---")
    # Convex combination soft-voting blend
    probs_blended = (alpha * probs_xgb) + ((1.0 - alpha) * probs_gat)

    final_preds = []
    triage_count = 0

    for i in range(len(probs_blended)):
        p = probs_blended[i]
       
        # Calculate conditional differential probability: P(B12 | IDA or B12)
        p_ida_b12_sum = p[1] + p[2]
        if p_ida_b12_sum > 0:
            p_b12_conditional = p[2] / p_ida_b12_sum
        else:
            p_b12_conditional = 0.5

        # Decision Cascade Triage
        if p[0] >= 0.50:
            final_preds.append(0)  # No Anemia
        elif p[3] >= 0.40:
            final_preds.append(3)  # Folate Deficiency
        elif lower_bound <= p_b12_conditional <= upper_bound:
            final_preds.append(4)  # GREY ZONE (Triage Bucket)
            triage_count += 1
        elif p_b12_conditional > upper_bound:
            final_preds.append(2)  # B12 Deficiency
        else:
            final_preds.append(1)  # IDA


    # --- UPDATE THIS AT THE VERY END OF run_full_hybrid_pipeline() ---
    final_preds = np.array(final_preds)
    target_names_with_grey = ETIOLOGY_NAMES + ['Grey Zone Triage']

    return gnn_model, test_graph, y_test, final_preds, target_names_with_grey

    # --- ADD THIS RETURN STATEMENT AT THE VERY END OF THE FUNCTION ---
    return y_test, final_preds, target_names_with_grey

    # Evaluate non-triaged high-confidence cases
    confident_mask = (final_preds != 4)
    acc = accuracy_score(y_test[confident_mask], final_preds[confident_mask])

    print("=" * 65)
    print(" HYBRID CASCADE + HETERO-GNN CLINICAL REPORT")
    print("=" * 65)
    print(f"Total Test Cohort Size       : {len(y_test)}")
    print(f"Triaged to Grey Zone (Code 4): {triage_count} ({triage_count / len(y_test) * 100:.2f}%)")
    print(f"High-Confidence Accuracy     : {acc:.4f}\n")
   
    target_names_with_grey = ETIOLOGY_NAMES + ['Grey Zone Triage']
    print(classification_report(y_test, final_preds, target_names=target_names_with_grey, labels=[0, 1, 2, 3, 4], digits=4))
    print("=" * 65)
   
if __name__ == "__main__":
    run_full_hybrid_pipeline(alpha=0.50, lower_bound=0.25, upper_bound=0.75)