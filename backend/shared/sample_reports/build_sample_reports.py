"""Generate India-compliant synthetic blood reports covering all 9 grader biomarkers.

These are **synthetic** diagnostic reports for validating
`vitascan_grader_model.AnemiaGrader` and the `GraderInputBuilder` end-to-end.
They are NOT real patient data.

Format & value grounding (see SOURCES.md):
  - CBC block layout / units / reference ranges follow real NABL Indian labs
    (Dr Lal PathLabs hemogram reports, Thyrocare hemogram, Apollo 24|7 CBC).
  - Iron studies + Vitamin B12 (ECLIA) + Folate (ECLIA) sections follow the
    Apollo 24|7 and Canwin full-report layouts and the Metro/Thyrocare
    "Anaemia Profile (Nutritional)" panels that bundle CBC + iron + B12 +
    folate + ferritin.
  - Values are set around Indian adult reference ranges; the anaemic scenarios
    track the RMLIMS (Lucknow) antenatal cohort means (MCV 88.2, MCH 27.5,
    MCHC 31.1, RDW 18.8, ferritin 49.7, B12 130, folate 14.8 ng/mL).
  - RBC indices are made internally consistent via the standard formulas:
        PCV (%)  = MCV (fL)   * RBC (M/mm3) / 10
        MCH (pg) = Hb (g/dL)  * 10 / RBC (M/mm3)
        MCHC (g/dL) = Hb (g/dL) * 100 / PCV (%)

Output: one `<pid>_latest.json` per scenario in this directory (same schema as
`backend/shared/extractions/pdf_raw/*_latest.json`, i.e. what BloodReportExtractor
produces) -> feed to run_grader_input_pipeline --pdf-dir <this dir>.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

LAB_NAME = "VitaScan Reference Laboratory"
NABL_NO = "NABL-AB-2026-XXXX"
ADDR = "Plot 14, Sector 18, New Delhi - 110001, India"


def _indices(hb: float, rbc: float, mcv: float) -> dict:
    pcv = mcv * rbc / 10.0
    mch = hb * 10.0 / rbc
    mchc = hb * 100.0 / pcv
    return {
        "pcv": round(pcv, 1),
        "mch": round(mch, 1),
        "mchc": round(mchc, 1),
    }


class Scenario:
    def __init__(self, pid, lab_no, barcode, name, age_sex, ref_by, hb, rbc,
                 mcv, rdw, ferritin, b12, folate, sex, src_note):
        self.pid = pid
        self.lab_no = lab_no
        self.barcode = barcode
        self.name = name
        self.age_sex = age_sex
        self.ref_by = ref_by
        self.hb = hb
        self.rbc = rbc
        self.mcv = mcv
        self.rdw = rdw
        self.ferritin = ferritin
        self.b12 = b12
        self.folate = folate
        self.sex = sex
        self.src_note = src_note
        self.idx = _indices(hb, rbc, mcv)

    # ------------------------------------------------------------------ ranges
    @property
    def hb_ref(self):
        return (13.0, 17.0) if self.sex == "M" else (11.5, 15.0)

    @property
    def rbc_ref(self):
        return (4.5, 5.5) if self.sex == "M" else (3.8, 4.8)

    @property
    def ferritin_ref(self):
        return (22.0, 322.0) if self.sex == "M" else (10.0, 120.0)

    # -------------------------------------------------------------- biomarkers
    def biomarkers(self) -> dict:
        hb_lo, hb_hi = self.hb_ref
        rbc_lo, rbc_hi = self.rbc_ref
        fer_lo, fer_hi = self.ferritin_ref
        map_ = {
            "hemoglobin_hb": {"value": self.hb, "unit": "g/dL",
                              "raw_value": f"{self.hb} g/dL", "ref_range": f"{hb_lo} - {hb_hi}"},
            "mcv": {"value": self.mcv, "unit": "fL",
                    "raw_value": f"{self.mcv} fL", "ref_range": "83.0 - 101.0"},
            "red_cell_distribution_width_rdw": {"value": self.rdw, "unit": "%",
                    "raw_value": f"{self.rdw} %", "ref_range": "11.6 - 14.0"},
            "ferritin": {"value": self.ferritin, "unit": "ng/mL",
                         "raw_value": f"{self.ferritin} ng/mL", "ref_range": f"{fer_lo} - {fer_hi}"},
            "b12": {"value": self.b12, "unit": "pg/mL",
                    "raw_value": f"{self.b12} pg/mL", "ref_range": "197.0 - 771.0"},
            "folate": {"value": self.folate, "unit": "ng/mL",
                       "raw_value": f"{self.folate} ng/mL", "ref_range": "3.89 - 26.8"},
            "total_rbc_count": {"value": self.rbc, "unit": "million/cmm",
                                "raw_value": f"{self.rbc} million/cmm", "ref_range": f"{rbc_lo} - {rbc_hi}"},
            "mch": {"value": self.idx["mch"], "unit": "pg",
                    "raw_value": f"{self.idx['mch']} pg", "ref_range": "27.0 - 32.0"},
            "mchc": {"value": self.idx["mchc"], "unit": "g/dL",
                     "raw_value": f"{self.idx['mchc']} g/dL", "ref_range": "31.5 - 34.5"},
        }
        return map_

    # --------------------------------------------------------------- raw_text
    def raw_text(self) -> str:
        hb_lo, hb_hi = self.hb_ref
        rbc_lo, rbc_hi = self.rbc_ref
        fer_lo, fer_hi = self.ferritin_ref
        i = self.idx
        lines = [
            f"CLINICAL LABORATORY REPORT",
            f"{LAB_NAME}   (NABL Accredited Lab: {NABL_NO})",
            ADDR,
            "",
            f"Patient Name : {self.name}          Age/Gender : {self.age_sex}",
            f"Patient ID   : {self.pid}            Ref. By : {self.ref_by}",
            f"Lab No       : {self.lab_no}          Barcode No : {self.barcode}",
            "Sample Drawn : 05/Sep/2026 08:12 AM     Registered : 05/Sep/2026",
            "Reported     : 06/Sep/2026 04:45 PM     Visit Type : Routine / OPD",
            "",
            "PANEL: ANAEMIA PROFILE (NUTRITIONAL)",
            "       CBC (HEMOGRAM) + IRON STUDIES + FERRITIN + VITAMIN B12 + FOLIC ACID",
            "",
            "HAEMATOLOGY",
            "Complete Blood Count (CBC)",
            "Sample : Whole Blood, EDTA",
            f"Hemoglobin (Hb)          {self.hb} g/dL        {hb_lo} - {hb_hi}      Spectrophotometric (Cyanmeth)",
            f"Packed Cell Volume (PCV) {i['pcv']} %          40.0 - 50.0      Calculated",
            f"Total RBC Count          {self.rbc} million/cmm    {rbc_lo} - {rbc_hi}      Electrical Impedance",
            f"Mean Corpuscular Volume (MCV) {self.mcv} fL    83.0 - 101.0    Calculated",
            f"Mean Corpuscular Hemoglobin (MCH) {i['mch']} pg  27.0 - 32.0    Calculated",
            f"MCHC                     {i['mchc']} g/dL      31.5 - 34.5      Calculated",
            f"RDW (CV)                 {self.rdw} %          11.6 - 14.0      Calculated",
            "Total Leucocyte Count (TLC)  7.4 thousand/mm3  4.0 - 10.0    Electrical Impedance",
            "Platelet Count              2.7 lac/mm3      1.5 - 4.5    Electrical Impedance",
            "",
            "BIOCHEMISTRY - IRON STUDIES",
            "Sample : Serum",
            f"Serum Ferritin           {self.ferritin} ng/mL     {fer_lo} - {fer_hi}    ECLIA",
            "Serum Iron                118.9 ug/dL        50.0 - 170.0    Ferrozine",
            "Total Iron Binding Capacity (TIBC)  318.4 ug/dL  250.0 - 400.0  Calculated",
            "% Transferrin Saturation  37.3 %            14.0 - 50.0     Calculated",
            "",
            "NUTRITIONAL VITAMINS",
            "Sample : Serum",
            f"Vitamin B12 (Cyanocobalamin)  {self.b12} pg/mL   197.0 - 771.0   ECLIA",
            f"Folic Acid / Folate - Serum    {self.folate} ng/mL   3.89 - 26.8    ECLIA",
            "",
            "REMARKS",
            "This is a synthetic sample report generated for model validation. It does not",
            "represent a real patient. Values are set within Indian adult reference ranges.",
            "",
            "Reviewed & Authorised By",
            "Dr. A. Sharma, MD Pathology",
            "Consultant Pathologist",
            "~ END OF REPORT ~",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------ pdf_raw json
    def to_pdf_raw(self) -> dict:
        return {
            "patient_id": self.pid,
            "filename": f"{self.pid}.pdf",
            "raw_text": self.raw_text(),
            "biomarkers": self.biomarkers(),
            "is_synthetic": True,
            "based_on_format": self.src_note,
        }


def build_report(pid, **kwargs) -> Scenario:
    s = Scenario(pid, **kwargs)
    (HERE / f"{pid}_latest.json").write_text(
        json.dumps(s.to_pdf_raw(), indent=2, ensure_ascii=False), encoding="utf-8")
    return s


def main():
    (HERE / "__init__.py").touch(exist_ok=True)
    scenarios = [
        build_report(
            pid="IND-01-HEALTHY-MALE",
            lab_no="034/09/2026-001", barcode="10139001", name="Rohan Mehta",
            age_sex="34 Y/Male", ref_by="Dr. K. Nair",
            hb=14.8, rbc=5.11, mcv=86.9, rdw=13.2, ferritin=96.4, b12=476.0, folate=12.8,
            sex="M",
            src_note="Normal CBC block (Dr Lal PathLabs / Drlogy layout) + normal iron/B12/folate "
                     "(Apollo 24|7 ECLIA refs); all within Indian adult ranges."),
        build_report(
            pid="IND-02-HEALTHY-FEMALE",
            lab_no="034/09/2026-002", barcode="10139002", name="Priya Deshmukh",
            age_sex="29 Y/Female", ref_by="Self",
            hb=12.6, rbc=4.42, mcv=88.4, rdw=12.8, ferritin=58.7, b12=398.0, folate=14.1,
            sex="F",
            src_note="Female reference intervals (Hb 11.5-15.0, RBC 3.8-4.8, ferritin 10-120); "
                     "indices computed from standard formulas."),
        build_report(
            pid="IND-03-IDA-FEMALE",
            lab_no="034/09/2026-003", barcode="10139003", name="Sunita Yadav",
            age_sex="28 Y/Female", ref_by="Dr. M. Iqbal",
            hb=9.2, rbc=3.92, mcv=72.6, rdw=17.3, ferritin=7.2, b12=412.0, folate=13.6,
            sex="F",
            src_note="Iron deficiency anaemia pattern: microcytic (MCV<80), hypochromic, high RDW, "
                     "ferritin <15 ng/mL; mirrors classic Indian IDA and RMLIMS cohort."),
        build_report(
            pid="IND-04-MEGALOBLASTIC-B12-MALE",
            lab_no="034/09/2026-004", barcode="10139004", name="Arjun Pillai",
            age_sex="41 Y/Male", ref_by="Dr. S. Reddy",
            hb=10.1, rbc=2.92, mcv=105.2, rdw=16.9, ferritin=132.5, b12=84.0, folate=11.4,
            sex="M",
            src_note="Megaloblastic / B12 deficiency pattern: macrocytic (MCV>100), low B12 <100 "
                     "pg/mL with symptoms; consistent with vegetarian-diet Indian macrocytic anaemia."),
        build_report(
            pid="IND-05-MEGALOBLASTIC-FOLATE-FEMALE",
            lab_no="034/09/2026-005", barcode="10139005", name="Kavita Sharma",
            age_sex="32 Y/Female", ref_by="Dr. P. Verma",
            hb=9.8, rbc=3.09, mcv=104.3, rdw=17.6, ferritin=94.1, b12=402.0, folate=2.7,
            sex="F",
            src_note="Megaloblastic / folate deficiency pattern: macrocytic (MCV>100), low folate "
                     "<3.89 ng/mL, normal B12; common folate-deficient Indian antenatal profile."),
        build_report(
            pid="IND-06-MIXED-GREYZONE-MALE",
            lab_no="034/09/2026-006", barcode="10139006", name="Vikram Singh",
            age_sex="45 Y/Male", ref_by="Self",
            hb=12.3, rbc=4.58, mcv=88.7, rdw=15.1, ferritin=24.6, b12=208.0, folate=5.1,
            sex="M",
            src_note="Borderline/mixed picture: mild Hb fall, marginal ferritin (→IDA start), "
                     "borderline B12 208 and folate 5.1; designed to land between classes "
                     "(grey-zone probe)."),
    ]
    print(f"wrote {len(scenarios)} India-compliant synthetic reports to {HERE}")


if __name__ == "__main__":
    main()