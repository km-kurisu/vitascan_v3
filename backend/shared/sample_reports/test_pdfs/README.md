# Uploadable test blood reports

Real, text-layer PDFs of synthetic anaemia profiles. Upload any of them through
**Upload Scan → blood report** (or `POST /upload-report`) to exercise the whole
Path B chain: B1 extraction → B2 normalizer/grader input → B3 grader (XGB cascade +
hetero-GNN blend) → Mod C results page (including the *Anemia Grader Verdict* card).

All values are synthetic. No real patient data.

## Files

| PDF | Case | Expected verdict |
| --- | --- | --- |
| `IND-01-HEALTHY-MALE.pdf` | normal male, all 9 features present | No Anemia |
| `IND-02-HEALTHY-FEMALE.pdf` | normal female | No Anemia |
| `IND-03-IDA-FEMALE.pdf` | microcytic, ferritin 7.2 ng/mL | IDA |
| `IND-04-MEGALOBLASTIC-B12-MALE.pdf` | macrocytic (MCV 105), B12 84 pg/mL | Grey Zone Triage |
| `IND-05-MEGALOBLASTIC-FOLATE-FEMALE.pdf` | macrocytic, folate 2.7 ng/mL | Folate Deficiency |
| `IND-06-MIXED-GREYZONE-MALE.pdf` | borderline everything (grey-zone probe) | Grey Zone Triage |
| `IND-07-CBC-ONLY-FEMALE.pdf` | CBC only, **no** ferritin/B12/folate | IDA, 3 features KNN-imputed |

`MANIFEST.json` holds the exact value of all 9 grader features per file plus the
expected verdict/decision code, so a run can be diffed automatically.

`IND-04` and `IND-06` land in *Grey Zone Triage* by design — that is the model's
own triage threshold (`p_b12_cond` between the lower/upper bounds), not a
misclassification; the blended probabilities still lean B12 / No Anemia.

`IND-07` exists to exercise the imputation path: the results page then shows the
amber "6/9 biomarkers used · 3 biomarker(s) KNN-imputed" footer.

## Regenerating

```bash
backend/venv/bin/python backend/shared/sample_reports/build_test_pdfs.py
```

`build_sample_reports.py` owns the scenario values (also used for the
`IND-0X_*_latest.json` `pdf_raw` fixtures); this script only renders them as PDFs
via headless Chromium, so the two never drift.

## Layout constraints (do not "fix" these)

The report is laid out as one analyte per **single text line** (monospace block)
because the extractors are line-oriented:

* B1's generic scanner matches `<name> <value> <unit> <ref>` on one line. Analytes
  laid out in table *cells* come back from PyMuPDF as several lines and are missed.
* B1/B2 fall back to "first plausible number after the analyte name" when the
  direct pattern misses. So nothing else in the document may put a number within a
  few lines after an analyte word — that is why the printed patient ID is an
  accession (`VS2026-004`, not `IND-04-MEGALOBLASTIC-B12-MALE`) and why the panel
  header and remarks carry no analyte names or values. Getting this wrong silently
  produced B12 = the patient's age and B12 = the barcode.
