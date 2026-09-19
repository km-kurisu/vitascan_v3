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


class ModB3Output(BaseModel):
    patient_id: str
    severity: Dict[str, SeverityDetail] = Field(
        ...,
        description="Dynamic deficiency map (e.g. iron, b12, folate, anemia)"
    )
    attention_weights: Dict[str, Dict[str, float]] = Field(
        ...,
        description="GAT attention weights mapping deficiency -> biomarker -> weight"
    )
    baseline_comparison: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Baseline model evaluation scores (gat, logistic_regression, random_forest, xgboost)"
    )
    model_confidence: float = Field(..., ge=0.0, le=1.0)


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
    model_confidence: float = Field(..., ge=0.0, le=1.0)


# ==========================================
# Mod C Frontend Output Contract (Mod C -> Frontend)
# ==========================================

class PatientInfo(BaseModel):
    patient_id: str
    age: int
    gender: str


class KeyContributor(BaseModel):
    biomarker: str
    impact_pct: float
    direction: str = Field("negative", description="positive or negative impact on severity")


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


class SummaryInfo(BaseModel):
    flagged_deficiency_count: int
    overall_risk_band: str


class ModCFrontendOutput(BaseModel):
    generated_at: str
    schema_version: str = "1.0"
    patient: PatientInfo
    deficiencies: List[DeficiencyItem]
    summary: SummaryInfo


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
