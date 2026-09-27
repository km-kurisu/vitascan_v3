from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.shared.schemas import (
    ModCFrontendOutput,
    ModB3Output,
    ModA3Output,
    PatientMetadata,
    DeficiencyDetail,
    SummarySection,
    KeyContributor,
    CrosscheckIndicator,
    DietRecommendation,
    BloodParameter,
    UploadedReportInfo,
)
from backend.mod_c_explainer.explainer import LLMExplainer
from backend.mod_c_explainer.payload_helpers import (
    _display_name,
    build_blood_parameters,
    resolve_patient,
)
from backend.shared.storage import save_json_artifact

# Anemia grader decision -> the nutrient deficiency that decision implicates.
VERDICT_DEFICIENCY = {1: "iron", 2: "b12", 3: "folate"}
# Etiology probability label -> the same mapping, for grey-zone candidates.
ETIOLOGY_DEFICIENCY = {
    "IDA": "iron",
    "B12 Deficiency": "b12",
    "Folate Deficiency": "folate",
}

# Which extracted biomarkers explain each deficiency (first match wins order).
DEFICIENCY_MARKERS = {
    "iron": ("ferritin", "hemoglobin", "tibc", "iron", "mcv"),
    "b12": ("b12", "mcv", "rdw"),
    "folate": ("folate", "mcv", "rdw"),
    "vitamin_d": ("vitamin_d", "calcium"),
    "zinc": ("zinc",),
    "calcium": ("calcium", "albumin"),
    "anemia": ("hemoglobin", "mcv", "rdw", "ferritin"),
}

BAND_ORDER = {"none": 0, "mild": 1, "borderline": 1, "moderate": 2, "severe": 3}
COLOR_MAP = {"none": "#10b981", "mild": "#3b82f6", "borderline": "#3b82f6", "moderate": "#f59e0b", "severe": "#ef4444"}


class ModCFormatter:
    def __init__(self):
        self.explainer = LLMExplainer()

    # ---------- contributors ----------

    def _contributors(
        self,
        d_type: str,
        attention: Dict[str, float],
        parameters: List[BloodParameter],
        limit: int = 3,
    ) -> List[KeyContributor]:
        """Real biomarkers behind the card: GAT attention weights when the
        grader published them, else the largest measured deviations."""
        by_key = {p.key: p for p in parameters}

        if attention:
            ranked = sorted(attention.items(), key=lambda kv: kv[1], reverse=True)[:limit]
            contributors = []
            for k, w in ranked:
                # Prefer the table's own name; otherwise the reference name for that
                # biomarker so labels stay consistent when it wasn't measured here.
                key = str(k).lower()
                name = by_key[key].name if key in by_key else _display_name(key, {})
                contributors.append(
                    KeyContributor(
                        biomarker=name,
                        impact_pct=round(float(w) * 100, 1),
                        direction="negative",
                    )
                )
            return contributors

        contributors: List[KeyContributor] = []
        for marker in DEFICIENCY_MARKERS.get(d_type, ()):
            param = by_key.get(marker)
            if param is None or not param.deviation:
                continue
            contributors.append(
                KeyContributor(
                    biomarker=param.name,
                    impact_pct=round(min(abs(param.deviation) * 100, 99.0), 1),
                    direction="negative" if param.deviation < 0 else "positive",
                )
            )
            if len(contributors) >= limit:
                break
        return contributors

    @staticmethod
    def _measured_digest(
        d_type: str, parameters: List[BloodParameter]
    ) -> Dict[str, Dict[str, str]]:
        """The actual measured values for this deficiency's markers.

        Handed to the LLM explainer so it cites the patient's numbers; it used
        to receive a bare severity score and invent plausible lab values.
        """
        by_key = {p.key: p for p in parameters}
        digest: Dict[str, Dict[str, str]] = {}
        for marker in DEFICIENCY_MARKERS.get(d_type, ()):
            param = by_key.get(marker)
            if param is None:
                continue
            digest[marker] = {
                "name": param.name,
                "value": f"{param.value} {param.unit}".strip(),
                "range": param.normal_range,
                "status": param.status,
            }
        return digest

    # ---------- deficiency cards ----------

    def _verdict_deficiencies(
        self,
        mod_b3: ModB3Output,
        parameters: List[BloodParameter],
    ) -> List[DeficiencyDetail]:
        """Cards that follow the anemia grader verdict.

        The rule-based severity map disagrees with the verdict often enough to
        be misleading (it flagged "anemia: severe" on a report the grader
        called healthy), so once a verdict exists it is the source of truth:
        confirmed etiologies become confirmed cards, and anything the verdict
        does not implicate is dropped or marked unconfirmed.
        """
        model = mod_b3.model
        severity = mod_b3.severity
        attention = mod_b3.attention_weights or {}
        confidence = mod_b3.model_confidence
        cards: List[DeficiencyDetail] = []
        seen = set()

        def _band_for(d_type: str) -> tuple:
            item = severity.get(d_type)
            if item is not None and item.band != "none":
                return item.band, round(item.score, 3)
            return "none", 0.0

        def _add(d_type: str, band: str, score: float, confirmed: bool, etiology: Optional[str]) -> None:
            if d_type in seen:
                return
            seen.add(d_type)
            crosscheck = CrosscheckIndicator(available=False, agrees=False, source="none")
            cards.append(
                DeficiencyDetail(
                    type=d_type,
                    severity={
                        "band": band,
                        "score_pct": f"{int(round(score * 100))}%",
                        "badge_color": COLOR_MAP.get(band, "#3b82f6"),
                    },
                    explanation=self.explainer.explain_deficiency(
                        d_type, band, self._measured_digest(d_type, parameters)
                    ),
                    key_contributors=self._contributors(d_type, attention.get(d_type, {}), parameters),
                    crosscheck=crosscheck,
                    diet_recommendations=[
                        DietRecommendation(
                            suggestion=f"Increase consumption of {d_type}-rich foods and follow FSSAI RDA guidelines."
                        )
                    ],
                    model_confirmed=confirmed,
                    etiology=etiology,
                    model_confidence=round(confidence, 3) if confirmed else None,
                )
            )

        decision = int(model.decision_code)

        if decision in VERDICT_DEFICIENCY:
            d_type = VERDICT_DEFICIENCY[decision]
            band, score = _band_for(d_type)
            if band == "none":
                band, score = "moderate", 0.72
            _add(d_type, band, score, True, model.etiology)
        elif decision == 4:
            # Grey zone: no confirmed etiology. Show the leading candidates so
            # the section is not empty, flagged as needing review.
            proba = {k: v for k, v in (model.proba or {}).items() if k != "No Anemia"}
            for etiology, _p in sorted(proba.items(), key=lambda kv: kv[1], reverse=True)[:2]:
                d_type = ETIOLOGY_DEFICIENCY.get(etiology)
                if not d_type:
                    continue
                band, score = _band_for(d_type)
                if band == "none":
                    band, score = "mild", 0.55
                _add(d_type, band, score, False, etiology)

        # Biomarker-flagged deficiencies the verdict did not implicate stay
        # visible, but unconfirmed. "anemia" is a condition rather than a
        # deficiency, and flagging it is what made healthy reports look broken.
        for d_type, item in severity.items():
            if d_type in seen or d_type == "anemia" or item.band == "none":
                continue
            if d_type not in DEFICIENCY_MARKERS:
                continue
            _add(d_type, item.band, round(item.score, 3), False, None)

        return cards

    def _legacy_deficiencies(
        self, mod_b3: ModB3Output, parameters: List[BloodParameter]
    ) -> List[DeficiencyDetail]:
        """Pre-verdict behaviour: one card per severity entry."""
        cards: List[DeficiencyDetail] = []
        attention = mod_b3.attention_weights or {}
        for d_type, item in mod_b3.severity.items():
            cards.append(
                DeficiencyDetail(
                    type=d_type,
                    severity={
                        "band": item.band,
                        "score_pct": f"{int(round(item.score * 100))}%",
                        "badge_color": COLOR_MAP.get(item.band, "#3b82f6"),
                    },
                    explanation=self.explainer.explain_deficiency(
                        d_type, item.band, self._measured_digest(d_type, parameters)
                    ),
                    key_contributors=self._contributors(d_type, attention.get(d_type, {}), parameters),
                    crosscheck=CrosscheckIndicator(available=False, agrees=False, source="none"),
                    diet_recommendations=[
                        DietRecommendation(
                            suggestion=f"Increase consumption of {d_type}-rich foods and follow FSSAI RDA guidelines."
                        )
                    ],
                )
            )
        return cards

    # ---------- entry point ----------

    def format_pipeline_output(
        self,
        mod_b3: ModB3Output,
        mod_a3: Optional[ModA3Output] = None,
        extraction: Optional[Dict[str, Any]] = None,
        normalized_biomarkers: Optional[Dict[str, Any]] = None,
        uploaded_report: Optional[Dict[str, Any]] = None,
        model_input: Optional[Dict[str, Any]] = None,
    ) -> ModCFrontendOutput:
        patient_id = mod_b3.patient_id
        parameters = build_blood_parameters(extraction, normalized_biomarkers, model_input)

        if mod_b3.model is not None:
            deficiencies = self._verdict_deficiencies(mod_b3, parameters)
        else:
            deficiencies = self._legacy_deficiencies(mod_b3, parameters)

        if mod_a3 and mod_a3.crosscheck_signal:
            for card in deficiencies:
                signal = mod_a3.crosscheck_signal.get(card.type)
                if signal is not None:
                    card.crosscheck = CrosscheckIndicator(
                        available=True,
                        agrees=signal.agrees_with_path_b,
                        source=signal.source,
                    )

        flagged = [c for c in deficiencies if c.severity.get("band") != "none"]
        worst_band = "none"
        for card in flagged:
            band = str(card.severity.get("band", "none"))
            if BAND_ORDER.get(band, 0) > BAND_ORDER.get(worst_band, 0):
                worst_band = band

        report_info: Optional[UploadedReportInfo] = None
        if uploaded_report:
            report_info = UploadedReportInfo(**uploaded_report)
        elif extraction:
            report_info = UploadedReportInfo(
                filename=str(extraction.get("filename") or "report.pdf"),
                uploaded_at=datetime.now(timezone.utc).isoformat(),
                parse_confidence=extraction.get("parse_confidence"),
            )

        patient = resolve_patient(extraction, patient_id)

        output = ModCFrontendOutput(
            generated_at=datetime.now(timezone.utc).isoformat(),
            schema_version="1.1",
            patient=PatientMetadata(patient_id=patient.patient_id, age=patient.age, gender=patient.gender),
            deficiencies=deficiencies,
            summary=SummarySection(flagged_deficiency_count=len(flagged), overall_risk_band=worst_band),
            model=mod_b3.model,
            model_confidence=mod_b3.model_confidence,
            blood_parameters=parameters,
            uploaded_report=report_info,
        )

        save_json_artifact("results", patient_id, output.model_dump())
        return output
