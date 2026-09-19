import sys, os, io
sys.path.insert(0, r"I:\Code\Vitascan_V3_Gq")
os.chdir(r"I:\Code\Vitascan_V3_Gq")

from backend.path_b_blood_report.mod_b1_extractor.extractor import BloodReportExtractor
from backend.path_b_blood_report.mod_b1_extractor.biobert_extractor import BioBERTBiomarkerExtractor
from backend.path_b_blood_report.mod_b2_normalizer.normalizer import BiomarkerNormalizer
from backend.path_b_blood_report.mod_b3_grader.grader import PathBGrader
from PIL import Image, ImageDraw

print("1. Initializing Extractors & Models...")
ext = BloodReportExtractor()
bio = BioBERTBiomarkerExtractor()
norm = BiomarkerNormalizer()
grader = PathBGrader()

print(f"   Tesseract Executable Path: {ext.tesseract_cmd}")

print("\n2. Generating Sample Blood Report Image for OCR...")
img = Image.new('RGB', (800, 250), color=(255, 255, 255))
draw = ImageDraw.Draw(img)
report_text = (
    "COMPLETE BLOOD COUNT (CBC) & METABOLIC REPORT\n"
    "Patient: Test Patient | Gender: Female\n"
    "Hemoglobin: 10.4 g/dL\n"
    "Serum Ferritin: 11.2 ng/mL\n"
    "Vitamin B12: 210 pg/mL\n"
    "Serum Folate: 4.8 ng/mL"
)
draw.text((20, 20), report_text, fill=(0, 0, 0))

buf = io.BytesIO()
img.save(buf, format='PNG')
image_bytes = buf.getvalue()

print("\n3. Executing Tesseract OCR Extraction...")
raw_res = ext.extract_from_bytes(image_bytes, "sample_report.png", "PAT-TEST101")
print(f"   Method Used: {raw_res.get('extraction_method')}")
print("   OCR Extracted Text Output:")
print("-" * 50)
print(raw_res.get("raw_text"))
print("-" * 50)

print("\n4. Executing Biomarker Entity Recognition & Normalization...")
ner_res = bio.extract_biomarkers(raw_res["raw_text"])
print(f"   NER Extracted Biomarkers: {list(ner_res.get('biomarkers', {}).keys())}")

norm_res = norm.normalize(ner_res["biomarkers"])
print("   Normalized Biomarker Deviations:")
for k, v in norm_res.items():
    print(f"   - {k}: raw={v['raw_value']}, dev={v['deviation']}, abnormal={v['is_abnormal']}")

print("\n5. Executing Path B Severity Grading...")
b3_out = grader.grade("PAT-TEST101", norm_res)
print("   Path B Severity Output:")
for k, v in b3_out.severity.items():
    print(f"   - {k}: band={v.band}, score={v.score}, model={v.model}")

print("\n=== TESSERACT OCR TEST RUN PASSED SUCCESSFULLY ===")
