"""
Pydantic schemas matching docs/schemas/*.json specifications.
Source of truth for all backend module data exchange contracts.
"""
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


# ==========================================
# Mod B3 Output Contract (Path B -> Mod C)
# ==========================================

class SeverityDetail(BaseModel):
    score: float = Field(..., description="Severity score between 0.0 and 1.0")
    band: Literal["none", "mild", "moderate", "severe"] = Field(..., description="Severity classification band")
    model: str = Field(..., description="Model identifier used for inference (gat or baseline)")


class FeatureAttribution(BaseModel):
    """One biomarker's share of the explanation for a model verdict.

    ``share_pct`` is signed and expressed as a percentage; the contributions on
    a card sum to 100 in absolute value. It is a *share*, not a probability --
    it says how much of the explanation this biomarker carries, never how likely
    the verdict is. Presenting it as a confidence would be a misreading.
    """

    feature: str = Field(..., description="Canonical model feature, e.g. ferritin")
    label: str = Field(..., description="Display name, e.g. Serum Ferritin")
    share_pct: float = Field(..., description="Signed share of the explanation, as a percentage")
    estimated: bool = Field(
        False, description="True when the underlying value was KNN-imputed, not measured"
    )
    is_derived: bool = Field(
        False, description="True for computed indices (Mentzer, Srivastava, cell Hb density)"
    )
    derived_from: List[str] = Field(
        default_factory=list, description="Measured values a derived index is computed from"
    )


class ModelExplanation(BaseModel):
    """Per-feature attribution for the deployed model's verdict on one patient.

    Produced by exact TreeSHAP over the XGBoost cascade plus Integrated
    Gradients over the hetero-GNN. See
    ``mod_b3_grader/explainability`` for what these numbers do and do not
    guarantee; ``caveats`` carries the limits through to the UI so they cannot
    be silently dropped on the way to a patient-facing screen.
    """

    explained_class: int = Field(..., ge=0, le=3, description="Etiology index being explained")
    explained_label: str = Field(..., description="Name of the etiology being explained")
    contributions: List[FeatureAttribution] = Field(default_factory=list)
    p_b12_conditional: Optional[float] = Field(
        None, description="p(B12 | IDA or B12); the quantity the grey-zone band tests"
    )
    grey_zone_band: Optional[List[float]] = Field(
        None, description="[lower, upper] band p_b12_conditional fell inside, when in triage"
    )
    gnn_available: bool = Field(
        False, description="False when only the XGB cascade could be explained"
    )
    caveats: List[str] = Field(
        default_factory=list, description="Limits of this explanation, for display"
    )


class CounterfactualTarget(BaseModel):
    """The nearest value of one biomarker at which the verdict changes.

    Univariate: every other biomarker is held fixed. Real biomarkers move
    together, so this is a statement about the model's decision boundary, not a
    course a patient could follow. ``caveats`` on the parent says so.
    """

    feature: str = Field(..., description="Canonical model feature, e.g. ferritin")
    label: str = Field(..., description="Display name, e.g. Ferritin")
    unit: str = Field("", description="Reporting unit, e.g. ng/mL")
    current_value: float = Field(..., description="The patient's measured value")
    target_value: float = Field(..., description="Value at which the verdict changes")
    direction: Literal["increase", "decrease"]
    to_etiology: str = Field(..., description="Verdict on the other side of the boundary")
    at_range_boundary: bool = Field(
        False,
        description=(
            "True when the boundary sits at the edge of the searched range, so "
            "the true crossing may lie beyond it"
        ),
    )


class CounterfactualExplanation(BaseModel):
    """What would have to change for this verdict to move.

    Distinct from ``ModelExplanation``, which decomposes the verdict the model
    already reached. This one describes verdicts it did not reach, so every
    entry was verified by re-running the model on the proposed row before being
    reported.
    """

    targets: List[CounterfactualTarget] = Field(
        default_factory=list, description="Nearest boundary first"
    )
    unreachable: List[str] = Field(
        default_factory=list,
        description="Biomarkers whose whole plausible range leaves the verdict unchanged",
    )
    refused: List[str] = Field(
        default_factory=list,
        description="Biomarkers not searched because the value was estimated, not measured",
    )
    caveats: List[str] = Field(default_factory=list)


class ModelGradeDetail(BaseModel):
    decision_code: int = Field(..., ge=0, le=4, description="0 No Anemia, 1 IDA, 2 B12, 3 Folate, 4 Grey Zone Triage")
    etiology: str = Field(..., description="Model decision label (DECISION_NAMES[decision_code])")
    na_count: int = Field(..., ge=0, le=9, description="Missing base biomarkers fed to the model (KNN-imputed)")
    complete_input: bool = Field(..., description="True when all 9 base biomarkers were present")
    proba: Dict[str, float] = Field(..., description="Blended probabilities over the 4 etiologies")
    proba_xgb: Dict[str, float] = Field(..., description="XGB cascade probabilities")
    proba_gnn: Dict[str, float] = Field(..., description="Hetero-GNN probabilities")
    alpha: float = Field(..., ge=0.0, le=1.0, description="Soft-vote blend weight on the XGB cascade")
    imputed_features: Optional[Dict[str, float]] = Field(
        None, description="12-feature row after KNN imputation (log only, matches training order)"
    )
    explanation: Optional[ModelExplanation] = Field(
        None,
        description=(
            "Real per-feature attribution for this verdict. Absent when "
            "explanation is disabled or SHAP is unavailable; callers must fall "
            "back rather than assume it is present."
        ),
    )
    counterfactuals: Optional[CounterfactualExplanation] = Field(
        None,
        description=(
            "What would move this verdict, by how much. Absent unless explicitly "
            "requested -- the search costs seconds per patient. Callers must "
            "handle its absence."
        ),
    )


class ModB3Output(BaseModel):
    patient_id: str
    severity: Dict[str, SeverityDetail] = Field(
        ...,
        description="Dynamic deficiency map (e.g. iron, b12, folate, anemia)"
    )
    attention_weights: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "DEPRECATED, no longer populated. This previously carried a hardcoded "
            "GAT attention literal -- the same three numbers for every patient -- "
            "which was presented in the UI as if it were derived from the report. "
            "Real attribution now lives on `model.explanation`. Kept optional so "
            "older saved artifacts still parse; do not read it in new code."
        ),
    )
    baseline_comparison: Dict[str, Dict[str, float]] = Field(
        ...,
        description=(
            "Baseline model evaluation scores, keyed by deficiency then model "
            "name. Always empty today: no baseline has been evaluated. This "
            "previously carried a hardcoded literal that ignored its input and "
            "contradicted a second hardcoded copy in the frontend, so the pair "
            "read as a measured benchmark against the deployed model. Do not "
            "populate it from anywhere but a real held-out evaluation."
        ),
    )
    model_confidence: float = Field(..., ge=0.0, le=1.0)
    model: Optional[ModelGradeDetail] = Field(
        None,
        description="Deployed AnemiaGrader (xgb cascade + hetero-GNN) decision on the normalizer payload",
    )


# ==========================================
# Mod A3 Output Contract (Path A -> Mod C)
# ==========================================

class CrosscheckDetail(BaseModel):
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: Literal["nails", "eyes", "skin", "tongue", "hair"]
    agrees_with_path_b: bool


class ModA3Output(BaseModel):
    patient_id: str
    crosscheck_signal: Dict[str, CrosscheckDetail] = Field(
        ...,
        description="Optional cross-check signal mapping deficiency -> detail"
    )
    visual_findings: List[str] = Field(
        ...,
        description="Human-readable findings from the symptom image analysis"
    )
    model_confidence: float = Field(..., ge=0.0, le=1.0)


# ==========================================
# Mod C Frontend Output Contract (Mod C -> Frontend)
# ==========================================

class PatientInfo(BaseModel):
    patient_id: str
    age: Optional[int] = Field(None, description="Age in years when the report states one")
    gender: str = "Unknown"


class KeyContributor(BaseModel):
    biomarker: str
    impact_pct: float = Field(
        ..., description="Share of the explanation this biomarker carries, as a percentage"
    )
    direction: str = Field("negative", description="positive or negative impact on severity")
    estimated: bool = Field(
        False, description="True when the underlying value was KNN-imputed, not measured"
    )
    source: str = Field(
        "deviation",
        description=(
            "'model' when derived from real model attribution, 'deviation' when "
            "falling back to the largest measured deviations. These are not "
            "equivalent: only 'model' reflects what the model actually used."
        ),
    )


class FrontendCrosscheck(BaseModel):
    available: bool
    agrees: bool
    source: str


class DietRecommendation(BaseModel):
    suggestion: str
    fssai_checked: bool = True


class DeficiencyItem(BaseModel):
    type: str
    severity: Dict[str, str] = Field(
        ...,
        description="Contains band, score_pct, and badge_color"
    )
    explanation: str
    key_contributors: List[KeyContributor]
    crosscheck: FrontendCrosscheck
    diet_recommendations: List[DietRecommendation]
    model_confirmed: bool = Field(
        False,
        description="True when the anemia grader verdict implicates this deficiency",
    )
    etiology: Optional[str] = Field(
        None, description="Anemia grader verdict label behind this card (e.g. IDA)"
    )
    model_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Verdict confidence when model_confirmed is True"
    )


class BloodParameter(BaseModel):
    key: str = Field(..., description="Canonical biomarker key, e.g. hemoglobin")
    name: str = Field(..., description="Display name, e.g. Serum Ferritin")
    value: float
    unit: str = ""
    normal_range: str = Field(..., description="Reference interval as printed, e.g. 12.0 - 15.5")
    status: Literal["Low", "Normal", "High", "Borderline"] = "Normal"
    deviation: float = Field(0.0, description="Signed fractional deviation from the reference interval")
    model_input: bool = Field(
        False, description="True when this biomarker feeds the anemia grader"
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Extractor confidence")


class UploadedReportInfo(BaseModel):
    filename: str
    uploaded_at: str
    page_count: Optional[int] = None
    parse_confidence: Optional[float] = None


class SummaryInfo(BaseModel):
    flagged_deficiency_count: int
    overall_risk_band: str


class ModCFrontendOutput(BaseModel):
    generated_at: str
    schema_version: str = "1.0"
    patient: PatientInfo
    deficiencies: List[DeficiencyItem]
    summary: SummaryInfo
    model: Optional[ModelGradeDetail] = Field(
        None,
        description="Deployed AnemiaGrader (xgb cascade + hetero-GNN) verdict for the anemia workup",
    )
    model_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Peak blended probability from the deployed grader"
    )
    blood_parameters: List[BloodParameter] = Field(
        default_factory=list,
        description="Extracted key blood parameters with reference ranges and status",
    )
    uploaded_report: Optional[UploadedReportInfo] = Field(
        None, description="Metadata for the report the results were computed from"
    )


# Alias definitions for backwards compatibility
SeverityItem = SeverityDetail
CrosscheckItem = CrosscheckDetail


# Compatibility aliases
SeverityItem = SeverityDetail
CrosscheckItem = CrosscheckDetail
PatientMetadata = PatientInfo
DeficiencyDetail = DeficiencyItem
SummarySection = SummaryInfo
CrosscheckIndicator = FrontendCrosscheck
