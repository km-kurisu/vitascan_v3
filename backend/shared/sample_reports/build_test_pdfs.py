"""Render the synthetic anaemia scenarios as real, uploadable blood-report PDFs.

`build_sample_reports.py` emits the same six scenarios as `pdf_raw` JSON (what the
extractor produces). This script turns them into actual PDFs so the whole Path B
chain can be exercised from the UI: upload -> B1 extraction -> B2 normalizer ->
B3 grader -> Mod C results page.

Rendering uses headless Chromium (`--print-to-pdf`) on an HTML template, so the
output carries a real text layer (PyMuPDF extraction in B1 reads it directly, and
the vision/Tesseract fallbacks still work on the rasterised page).

An extra `IND-07-CBC-ONLY-*` scenario is included: a plain CBC with no ferritin /
B12 / folate, which exercises the grader's KNN-imputation path (the results page
shows an amber "N biomarker(s) KNN-imputed" footer).

Output: one `<pid>.pdf` per scenario in `test_pdfs/`, plus `MANIFEST.json` with the
expected grader verdict for each so results can be diffed after an upload.
"""
from __future__ import annotations

import html
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from build_sample_reports import (
    ADDR,
    LAB_NAME,
    NABL_NO,
    Scenario,
    scenarios,
)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "test_pdfs"

CHROME_CANDIDATES = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable")

# Reference intervals as printed in build_sample_reports.Scenario.raw_text.
MCV_REF = "83.0 - 101.0"
RDW_REF = "11.6 - 14.0"
MCH_REF = "27.0 - 32.0"
MCHC_REF = "31.5 - 34.5"
B12_REF = "197.0 - 771.0"
FOLATE_REF = "3.89 - 26.8"
IRON_REF = "50.0 - 170.0"
TIBC_REF = "250.0 - 400.0"
TSAT_REF = "14.0 - 50.0"
TLC_REF = "4.0 - 10.0"
PLT_REF = "1.5 - 4.5"

# Context analytes (not fed to the grader) — iron studies track the ferritin level.
EXTRA_ROWS: dict[str, dict[str, tuple[str, str, str]]] = {
    "IND-01-HEALTHY-MALE": {
        "tlc": ("7.4", "thousand/mm3", TLC_REF),
        "plt": ("2.7", PLT_REF),
        "iron": ("118.9", "ug/dL", IRON_REF),
        "tibc": ("318.4", "ug/dL", TIBC_REF),
        "tsat": ("37.3", "%", TSAT_REF),
    },
    "IND-02-HEALTHY-FEMALE": {
        "tlc": ("6.2", "thousand/mm3", TLC_REF),
        "plt": ("2.4", PLT_REF),
        "iron": ("96.5", "ug/dL", IRON_REF),
        "tibc": ("302.1", "ug/dL", TIBC_REF),
        "tsat": ("31.9", "%", TSAT_REF),
    },
    "IND-03-IDA-FEMALE": {
        "tlc": ("8.9", "thousand/mm3", TLC_REF),
        "plt": ("3.1", PLT_REF),
        "iron": ("28.4", "ug/dL", IRON_REF),
        "tibc": ("462.7", "ug/dL", TIBC_REF),
        "tsat": ("6.1", "%", TSAT_REF),
    },
    "IND-04-MEGALOBLASTIC-B12-MALE": {
        "tlc": ("5.6", "thousand/mm3", TLC_REF),
        "plt": ("1.9", PLT_REF),
        "iron": ("141.7", "ug/dL", IRON_REF),
        "tibc": ("286.3", "ug/dL", TIBC_REF),
        "tsat": ("49.5", "%", TSAT_REF),
    },
    "IND-05-MEGALOBLASTIC-FOLATE-FEMALE": {
        "tlc": ("6.8", "thousand/mm3", TLC_REF),
        "plt": ("2.2", PLT_REF),
        "iron": ("104.3", "ug/dL", IRON_REF),
        "tibc": ("311.8", "ug/dL", TIBC_REF),
        "tsat": ("33.4", "%", TSAT_REF),
    },
    "IND-06-MIXED-GREYZONE-MALE": {
        "tlc": ("7.1", "thousand/mm3", TLC_REF),
        "plt": ("2.5", PLT_REF),
        "iron": ("54.2", "ug/dL", IRON_REF),
        "tibc": ("401.6", "ug/dL", TIBC_REF),
        "tsat": ("13.5", "%", TSAT_REF),
    },
    "IND-07-CBC-ONLY-FEMALE": {
        "tlc": ("9.4", "thousand/mm3", TLC_REF),
        "plt": ("3.3", PLT_REF),
    },
}

# Grader verdict each scenario is expected to produce (from the composite India run).
EXPECTED = {
    "IND-01-HEALTHY-MALE": ("No Anemia", 0),
    "IND-02-HEALTHY-FEMALE": ("No Anemia", 0),
    "IND-03-IDA-FEMALE": ("IDA", 1),
    "IND-04-MEGALOBLASTIC-B12-MALE": ("Grey Zone Triage", 4),
    "IND-05-MEGALOBLASTIC-FOLATE-FEMALE": ("Folate Deficiency", 3),
    "IND-06-MIXED-GREYZONE-MALE": ("Grey Zone Triage", 4),
    "IND-07-CBC-ONLY-FEMALE": ("IDA", 1),
}

# The CBC-only probe: same microcytic picture as IND-03 but no iron/B12/folate,
# so the model has to KNN-impute 3 of its 9 features.
CBC_ONLY = dict(
    lab_no="034/09/2026-007",
    barcode="10139007",
    name="Meera Krishnan",
    age_sex="26 Y/Female",
    ref_by="Dr. R. Bhatt",
    hb=10.4,
    rbc=3.61,
    mcv=74.0,
    rdw=18.2,
    ferritin=None,
    b12=None,
    folate=None,
    sex="F",
    src_note="Plain CBC only (no ferritin/B12/folate) to exercise KNN imputation of the "
             "grader's 9-feature payload.",
)


REMARK = (
    "SYNTHETIC REPORT generated for VitaScan pipeline testing. Not a real patient "
    "and not for clinical use. Provenance for each scenario is recorded in MANIFEST.json."
)

# Monospace column widths for the analyte block (chars).
NAME_COL = 40
VAL_COL = 26


def _flag(value: float, ref: str) -> str:
    lo, hi = (float(x) for x in ref.split(" - "))
    if value < lo:
        return "L"
    if value > hi:
        return "H"
    return ""


def _rows(s: Scenario, include_chemistry: bool) -> list[dict]:
    hb_lo, hb_hi = s.hb_ref
    rbc_lo, rbc_hi = s.rbc_ref
    fer_lo, fer_hi = s.ferritin_ref
    i = s.idx
    b = s.biomarkers()
    extras = EXTRA_ROWS.get(s.pid, {})

    rows: list[tuple] = [
        ("Complete Blood Count (CBC)", None),
        ("Hemoglobin (Hb)", b["hemoglobin_hb"]["value"], "g/dL", f"{hb_lo} - {hb_hi}"),
        ("Packed Cell Volume (PCV)", i["pcv"], "%", "40.0 - 50.0"),
        ("Total RBC Count", s.rbc, "million/cmm", f"{rbc_lo} - {rbc_hi}"),
        ("Mean Corpuscular Volume (MCV)", s.mcv, "fL", MCV_REF),
        ("Mean Corpuscular Hemoglobin (MCH)", i["mch"], "pg", MCH_REF),
        ("MCHC (Mean Corpuscular Hb)", i["mchc"], "g/dL", MCHC_REF),
        ("RDW (CV)", s.rdw, "%", RDW_REF),
        ("Total Leucocyte Count (TLC)", float(extras["tlc"][0]), "thousand/mm3", TLC_REF),
        ("Platelet Count", float(extras["plt"][0]), "lac/mm3", PLT_REF),
    ]
    if include_chemistry:
        rows += [
            ("Iron Studies", None),
            ("Serum Ferritin", s.ferritin, "ng/mL", f"{fer_lo} - {fer_hi}"),
            ("Serum Iron", float(extras["iron"][0]), "ug/dL", IRON_REF),
            ("Total Iron Binding Capacity (TIBC)", float(extras["tibc"][0]), "ug/dL", TIBC_REF),
            ("Transferrin Saturation (%)", float(extras["tsat"][0]), "%", TSAT_REF),
            ("Nutritional Vitamins", None),
            ("Vitamin B12 (Cyanocobalamin)", s.b12, "pg/mL", B12_REF),
            ("Folic Acid (Folate) - Serum", s.folate, "ng/mL", FOLATE_REF),
        ]

    out: list[dict] = []
    for row in rows:
        if row[1] is None:
            out.append({"section": row[0]})
        else:
            name, value, unit, ref = row
            out.append({"name": name, "value": value, "unit": unit, "ref": ref})
    return out


def _table(s: Scenario, include_chemistry: bool) -> str:
    """One analyte per rendered text line, as a single monospace run.

    B1's parsers are line-oriented: the generic scanner matches
    ``<name> <value> <unit> <ref>`` on one line, and the alias patterns take the
    first number following an analyte name. So a row must be one unbroken text
    run (spans in separate cells come back from PyMuPDF as separate lines), and
    nothing else in the document may put a number right after an analyte name.
    """
    lines: list[str] = []
    for r in _rows(s, include_chemistry):
        if "section" in r:
            lines.append(f'\n<div class="sec">{html.escape(r["section"])}</div>')
            continue
        num = float(r["value"])
        flag = _flag(num, r["ref"])
        name = f'{r["name"]} '
        val = f'{r["value"]} {r["unit"]}'
        ref = f'({r["ref"]})'
        pad = max(1, NAME_COL - len(name))
        row = f"{name}{' ' * pad}{val}"
        row += f"{' ' * max(1, VAL_COL - len(val))}{ref}"
        if flag:
            row += f"   {flag}"
        lines.append(f'<div class="row">{html.escape(row)}</div>')
    return "".join(lines)


def _page(s: Scenario, include_chemistry: bool, remark: str) -> str:
    e = html.escape
    # Real labs print an accession number here, not the scenario name. Keeping the
    # printed ID free of analyte words matters: the NER's multiline fallback grabs
    # the first number after any short line containing an alias, so an ID like
    # "IND-04-MEGALOBLASTIC-B12-MALE" makes the following Barcode row the "B12".
    accession = f"VS2026-{s.lab_no.rsplit('-', 1)[-1]}"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{e(s.pid)}</title>
<style>
  @page {{ size: A4; margin: 14mm 12mm; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: "DejaVu Sans", Arial, sans-serif; color: #111; font-size: 11px; margin: 0; }}
  .head {{ border-bottom: 3px solid #1D61E7; padding-bottom: 8px; margin-bottom: 10px; }}
  .lab {{ font-size: 17px; font-weight: 800; color: #1D61E7; letter-spacing: -0.2px; }}
  .sub {{ font-size: 9.5px; color: #555; margin-top: 2px; }}
  h1 {{ font-size: 13px; text-align: center; margin: 12px 0 4px; text-transform: uppercase;
        letter-spacing: 1.4px; }}
  .panel {{ text-align: center; font-size: 10px; color: #444; margin-bottom: 10px; }}
  table.meta {{ width: 100%; border-collapse: collapse; margin-bottom: 12px; font-size: 10px; }}
  table.meta td {{ border: 1px solid #ccc; padding: 4px 6px; }}
  table.meta td.k {{ background: #f2f5fa; font-weight: 700; width: 17%; color: #333; }}
  .panel-hd {{ background: #e8eefb; border: 1px solid #b9c6e4; padding: 4px 8px; }}
  .panel-hd span {{ font-size: 9px; font-weight: 800; text-transform: uppercase;
                     letter-spacing: 0.5px; color: #24365c; }}
  .mono {{ font-family: "DejaVu Sans Mono", "Courier New", monospace; }}
  .row {{ font-size: 9.5px; white-space: pre; border: 1px solid #dcdcdc; border-top: none;
          padding: 3px 8px; }}
  .sec {{ background: #f7f8fa; font-weight: 800; font-size: 9.5px; text-transform: uppercase;
          letter-spacing: 0.8px; color: #1D61E7; border: 1px solid #dcdcdc; border-top: none;
          border-left: 3px solid #1D61E7; padding: 4px 8px; }}
  .remarks {{ margin-top: 12px; border: 1px solid #dcdcdc; background: #fafbfc; padding: 8px 10px;
              font-size: 9.5px; line-height: 1.55; color: #333; }}
  .foot {{ margin-top: 16px; display: flex; justify-content: space-between; align-items: flex-end;
           border-top: 1px solid #ccc; padding-top: 7px; font-size: 9.5px; color: #444; }}
  .sig {{ text-align: right; }} .sig b {{ font-size: 10.5px; color: #111; }}
</style></head><body>
  <div class="head">
    <div class="lab">{e(LAB_NAME)}</div>
    <div class="sub">NABL Accredited Laboratory: {e(NABL_NO)} &nbsp;|&nbsp; {e(ADDR)}</div>
  </div>

  <h1>Clinical Laboratory Report</h1>
  <div class="panel">Panel: {'Anaemia Profile (Nutritional) - see results below' if include_chemistry else 'Complete Blood Count (Haemogram)'}</div>

  <table class="meta">
    <tr><td class="k">Patient</td><td>{e(s.name)}</td><td class="k">Age / Gender</td><td>{e(s.age_sex)}</td></tr>
    <tr><td class="k">Patient ID</td><td>{e(accession)}</td><td class="k">Referred By</td><td>{e(s.ref_by)}</td></tr>
    <tr><td class="k">Lab No</td><td>{e(s.lab_no)}</td><td class="k">Barcode</td><td>{e(s.barcode)}</td></tr>
    <tr><td class="k">Sample</td><td>Whole Blood (EDTA) / Serum</td><td class="k">Visit Type</td><td>Routine / OPD</td></tr>
    <tr><td class="k">Collected</td><td>05/Sep/2026 08:12 AM</td><td class="k">Reported</td><td>06/Sep/2026 04:45 PM</td></tr>
  </table>

  <div class="panel-hd mono"><span>Test Name</span>
  <span>Result &amp; Unit</span> <span>Biological Reference Interval</span> <span>Flag</span></div>
  <div class="mono">{_table(s, include_chemistry)}</div>

  <div class="remarks"><b>Remarks:</b> {e(remark)}</div>

  <div class="foot">
    <div>{e(LAB_NAME)}<br/>NABL No: {e(NABL_NO)}</div>
    <div class="sig"><b>Dr. A. Sharma, MD Pathology</b><br/>Consultant Pathologist<br/>Electronically verified</div>
  </div>
</body></html>"""


def _chrome() -> str:
    for name in CHROME_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    raise SystemExit(
        "No Chromium/Chrome binary found (needed to render PDFs). "
        f"Install one or edit CHROME_CANDIDATES in {__file__}."
    )


def _to_pdf(chrome: str, page_html: str, out_pdf: Path) -> None:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "report.html"
        dst = Path(td) / "report.pdf"
        src.write_text(page_html, encoding="utf-8")
        proc = subprocess.run(
            [
                chrome,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                "--no-pdf-header-footer",
                f"--print-to-pdf={dst}",
                src.as_uri(),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if not dst.exists():
            raise SystemExit(f"chromium failed to render {out_pdf.name}:\n{proc.stderr[-2000:]}")
        shutil.copyfile(dst, out_pdf)


def main() -> None:
    chrome = _chrome()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_scenarios: list[tuple[Scenario, bool]] = [(s, True) for s in scenarios()]
    all_scenarios.append((Scenario(pid="IND-07-CBC-ONLY-FEMALE", **CBC_ONLY), False))

    manifest = []
    for s, include_chemistry in all_scenarios:
        if not include_chemistry:
            s.ferritin, s.b12, s.folate = 0.0, 0.0, 0.0
        out_pdf = OUT_DIR / f"{s.pid}.pdf"
        _to_pdf(chrome, _page(s, include_chemistry, REMARK), out_pdf)
        etiology, code = EXPECTED[s.pid]
        manifest.append({
            "patient_id": s.pid,
            "pdf": out_pdf.name,
            "age_gender": s.age_sex,
            "has_iron_studies": include_chemistry,
            "expected_etiology": etiology,
            "expected_decision_code": code,
            "biochemistry": {
                "hemoglobin": s.hb, "rbc": s.rbc, "mcv": s.mcv, "rdw": s.rdw,
                "mch": s.idx["mch"], "mchc": s.idx["mchc"], "pcv": s.idx["pcv"],
                "ferritin": s.ferritin if include_chemistry else None,
                "b12": s.b12 if include_chemistry else None,
                "folate": s.folate if include_chemistry else None,
            },
            "expected_na_count": 0 if include_chemistry else 3,
            "note": s.src_note,
        })
        print(f"wrote {out_pdf.name}  ({out_pdf.stat().st_size // 1024} KB)")

    (OUT_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{len(manifest)} PDFs + MANIFEST.json in {OUT_DIR}")


if __name__ == "__main__":
    main()
