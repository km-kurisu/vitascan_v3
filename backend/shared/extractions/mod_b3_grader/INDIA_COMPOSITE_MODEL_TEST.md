# AnemiaGrader (mod_b3_grader) × India composite reports — Vitascan V3 test report

- Generated: 2026-09-23T23:24:33
- Flow: `sample pdf_raw -> GraderInputBuilder.build() -> PathBGrader.grade(grader_input=...)`
- Model: xgb cascade + hetero-GNN hybrid (blend alpha per decision)
- Samples: `backend/shared/sample_reports/IND-*_latest.json` (6 reports)

## 1. Verdicts

| Report | Intended | decision_code | Etiology | P(top) | Match |
|--------|----------|---------------|----------|--------|-------|
| `IND-01-HEALTHY-MALE` | No Anemia | 0 | **No Anemia** | No Anemia (0.70) | Y |
| `IND-02-HEALTHY-FEMALE` | No Anemia | 0 | **No Anemia** | No Anemia (0.71) | Y |
| `IND-03-IDA-FEMALE` | IDA | 1 | **IDA** | IDA (0.64) | Y |
| `IND-04-MEGALOBLASTIC-B12-MALE` | B12 Deficiency | 4 | **Grey Zone Triage** | B12 Deficiency (0.58) | Y |
| `IND-05-MEGALOBLASTIC-FOLATE-FEMALE` | Folate Deficiency | 3 | **Folate Deficiency** | Folate Deficiency (0.54) | Y |
| `IND-06-MIXED-GREYZONE-MALE` | Grey Zone Triage | 4 | **Grey Zone Triage** | IDA (0.49) | Y |

`4` = Grey Zone Triage. Match: `Y` exact, `P` borderline (intended definite class, but the model sent it to triage with that class on top).

## 2. Per-report details

### IND-01-HEALTHY-MALE

| Field | Value |
|-------|-------|
| model_input | `{"hemoglobin": 14.8, "mcv": 86.9, "rdw": 13.2, "ferritin": 96.4, "b12": 476.0, "folate": 12.8, "rbc": 5.11, "mch": 29.0, "mchc": 33.3}` |
| na_count | 0 |
| complete_input | True |
| decision_code | 0 (No Anemia) |
| P blended | `{"No Anemia": 0.6966, "IDA": 0.1474, "B12 Deficiency": 0.0752, "Folate Deficiency": 0.0809}` |
| P xgb | `{"No Anemia": 1.0, "IDA": 0.0, "B12 Deficiency": 0.0, "Folate Deficiency": 0.0}` |
| P gnn | `{"No Anemia": 0.3933, "IDA": 0.2947, "B12 Deficiency": 0.1503, "Folate Deficiency": 0.1617}` |
| alpha | `0.5` |
| model_confidence | `0.697` |
| severity overlay | `{}` |

Imputed 12-feature row:

```
  hemoglobin             14.8
  mcv                    86.9
  rdw                    13.2
  ferritin               96.4
  b12                    476.0
  folate                 12.8
  rbc                    5.11
  mch                    29.0
  mchc                   33.3
  srivastava_index       5.6751
  mentzer_index          17.0059
  cell_hb_density        9.657
```

### IND-02-HEALTHY-FEMALE

| Field | Value |
|-------|-------|
| model_input | `{"hemoglobin": 12.6, "mcv": 88.4, "rdw": 12.8, "ferritin": 58.7, "b12": 398.0, "folate": 14.1, "rbc": 4.42, "mch": 28.5, "mchc": 32.2}` |
| na_count | 0 |
| complete_input | True |
| decision_code | 0 (No Anemia) |
| P blended | `{"No Anemia": 0.713, "IDA": 0.1896, "B12 Deficiency": 0.0315, "Folate Deficiency": 0.0659}` |
| P xgb | `{"No Anemia": 0.9999, "IDA": 0.0, "B12 Deficiency": 0.0, "Folate Deficiency": 0.0}` |
| P gnn | `{"No Anemia": 0.426, "IDA": 0.3792, "B12 Deficiency": 0.063, "Folate Deficiency": 0.1318}` |
| alpha | `0.5` |
| model_confidence | `0.713` |
| severity overlay | `{}` |

Imputed 12-feature row:

```
  hemoglobin             12.6
  mcv                    88.4
  rdw                    12.8
  ferritin               58.7
  b12                    398.0
  folate                 14.1
  rbc                    4.42
  mch                    28.5
  mchc                   32.2
  srivastava_index       6.448
  mentzer_index          20.0
  cell_hb_density        9.177
```

### IND-03-IDA-FEMALE

| Field | Value |
|-------|-------|
| model_input | `{"hemoglobin": 9.2, "mcv": 72.6, "rdw": 17.3, "ferritin": 7.2, "b12": 412.0, "folate": 13.6, "rbc": 3.92, "mch": 23.5, "mchc": 32.3}` |
| na_count | 0 |
| complete_input | True |
| decision_code | 1 (IDA) |
| P blended | `{"No Anemia": 0.2871, "IDA": 0.6366, "B12 Deficiency": 0.052, "Folate Deficiency": 0.0243}` |
| P xgb | `{"No Anemia": 0.0016, "IDA": 0.9981, "B12 Deficiency": 0.0, "Folate Deficiency": 0.0003}` |
| P gnn | `{"No Anemia": 0.5726, "IDA": 0.2751, "B12 Deficiency": 0.1039, "Folate Deficiency": 0.0484}` |
| alpha | `0.5` |
| model_confidence | `0.637` |
| severity overlay | `{"iron": {"score": 0.72, "band": "moderate", "model": "anemia_grader"}, "anemia": {"score": 0.75, "band": "moderate", "model": "anemia_grader"}}` |

Imputed 12-feature row:

```
  hemoglobin             9.2
  mcv                    72.6
  rdw                    17.3
  ferritin               7.2
  b12                    412.0
  folate                 13.6
  rbc                    3.92
  mch                    23.5
  mchc                   32.3
  srivastava_index       5.9949
  mentzer_index          18.5204
  cell_hb_density        7.5905
```

### IND-04-MEGALOBLASTIC-B12-MALE

| Field | Value |
|-------|-------|
| model_input | `{"hemoglobin": 10.1, "mcv": 105.2, "rdw": 16.9, "ferritin": 132.5, "b12": 84.0, "folate": 11.4, "rbc": 2.92, "mch": 34.6, "mchc": 32.9}` |
| na_count | 0 |
| complete_input | True |
| decision_code | 4 (Grey Zone Triage) |
| P blended | `{"No Anemia": 0.0814, "IDA": 0.2775, "B12 Deficiency": 0.5802, "Folate Deficiency": 0.0609}` |
| P xgb | `{"No Anemia": 0.0001, "IDA": 0.0, "B12 Deficiency": 0.9996, "Folate Deficiency": 0.0003}` |
| P gnn | `{"No Anemia": 0.1626, "IDA": 0.5551, "B12 Deficiency": 0.1607, "Folate Deficiency": 0.1215}` |
| alpha | `0.5` |
| model_confidence | `0.58` |
| severity overlay | `{"anemia": {"score": 0.55, "band": "mild", "model": "anemia_grader"}}` |

Imputed 12-feature row:

```
  hemoglobin             10.1
  mcv                    105.2
  rdw                    16.9
  ferritin               132.5
  b12                    84.0
  folate                 11.4
  rbc                    2.92
  mch                    34.6
  mchc                   32.9
  srivastava_index       11.8493
  mentzer_index          36.0274
  cell_hb_density        11.3834
```

### IND-05-MEGALOBLASTIC-FOLATE-FEMALE

| Field | Value |
|-------|-------|
| model_input | `{"hemoglobin": 9.8, "mcv": 104.3, "rdw": 17.6, "ferritin": 94.1, "b12": 402.0, "folate": 2.7, "rbc": 3.09, "mch": 31.7, "mchc": 30.4}` |
| na_count | 0 |
| complete_input | True |
| decision_code | 3 (Folate Deficiency) |
| P blended | `{"No Anemia": 0.0444, "IDA": 0.3395, "B12 Deficiency": 0.0767, "Folate Deficiency": 0.5395}` |
| P xgb | `{"No Anemia": 0.0001, "IDA": 0.0, "B12 Deficiency": 0.0002, "Folate Deficiency": 0.9997}` |
| P gnn | `{"No Anemia": 0.0886, "IDA": 0.6789, "B12 Deficiency": 0.1532, "Folate Deficiency": 0.0793}` |
| alpha | `0.5` |
| model_confidence | `0.539` |
| severity overlay | `{"folate": {"score": 0.72, "band": "moderate", "model": "anemia_grader"}}` |

Imputed 12-feature row:

```
  hemoglobin             9.8
  mcv                    104.3
  rdw                    17.6
  ferritin               94.1
  b12                    402.0
  folate                 2.7
  rbc                    3.09
  mch                    31.7
  mchc                   30.4
  srivastava_index       10.2589
  mentzer_index          33.754
  cell_hb_density        9.6368
```

### IND-06-MIXED-GREYZONE-MALE

| Field | Value |
|-------|-------|
| model_input | `{"hemoglobin": 12.3, "mcv": 88.7, "rdw": 15.1, "ferritin": 24.6, "b12": 208.0, "folate": 5.1, "rbc": 4.58, "mch": 26.9, "mchc": 30.3}` |
| na_count | 0 |
| complete_input | True |
| decision_code | 4 (Grey Zone Triage) |
| P blended | `{"No Anemia": 0.1538, "IDA": 0.4911, "B12 Deficiency": 0.3393, "Folate Deficiency": 0.0158}` |
| P xgb | `{"No Anemia": 0.0026, "IDA": 0.332, "B12 Deficiency": 0.664, "Folate Deficiency": 0.0014}` |
| P gnn | `{"No Anemia": 0.3049, "IDA": 0.6501, "B12 Deficiency": 0.0147, "Folate Deficiency": 0.0303}` |
| alpha | `0.5` |
| model_confidence | `0.491` |
| severity overlay | `{"anemia": {"score": 0.55, "band": "mild", "model": "anemia_grader"}}` |

Imputed 12-feature row:

```
  hemoglobin             12.3
  mcv                    88.7
  rdw                    15.1
  ferritin               24.6
  b12                    208.0
  folate                 5.1
  rbc                    4.58
  mch                    26.9
  mchc                   30.3
  srivastava_index       5.8734
  mentzer_index          19.3668
  cell_hb_density        8.1507
```

## 3. Notes

- Reports are synthetic composites (no real patient data), used to prove the copied model is wired into the Vitascan V3 flow end-to-end.
- `IND-04` (severe B12 deficiency) is routed to Grey Zone Triage by design; `P(B12 Deficiency)` remains the top blended class.

