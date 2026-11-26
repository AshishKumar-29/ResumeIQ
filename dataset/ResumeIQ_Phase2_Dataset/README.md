# ResumeIQ – Phase 2 Synthetic Resume Dataset

## Purpose
This starter dataset was created for the Resume Information Extraction capstone project. It contains 60 fully synthetic English resume examples. It contains no real candidate records.

## Contents
- `resumes_txt/`: 60 canonical plain-text resume files.
- `sample_uploads/pdf/`: 6 synthetic PDF examples for GUI upload testing.
- `sample_uploads/docx/`: 6 synthetic DOCX examples for GUI upload testing.
- `annotations/all_annotations.jsonl`: all character-level entity annotations and split assignments.
- `annotations/train.jsonl`, `validation.jsonl`, `test.jsonl`: split annotations.
- `annotations/resume_manifest.csv`: per-resume metadata and entity counts.
- `spacy/train.spacy`, `validation.spacy`, `test.spacy`: spaCy v3 DocBin datasets.

## Split
- Train: 42 resumes (70%)
- Validation: 9 resumes (15%)
- Test: 9 resumes (15%)
The split is deterministic (seed 20261125). Keep the test set out of model training and model-selection decisions.

## Entity labels
- `PERSON`: candidate name
- `EMAIL`: email address
- `PHONE`: phone number (synthetic, deliberately non-dialable placeholder)
- `SKILL`: one technical or professional skill phrase
- `DEGREE`: education qualification
- `INSTITUTION`: college/university/institute name
- `JOB_TITLE`: role/designation
- `COMPANY`: employer/company name
- `DATE`: education or employment date/range
- `DURATION`: experience duration

## Annotation format
JSONL fields: `id`, `text`, `entities`, and (in the all-annotations file) `split`. Each entity is `[start, end, label]`, where start is inclusive, end is exclusive, and offsets count Unicode characters in the exact canonical text.

## Important limitations
1. All resumes are synthetic template-generated examples; they do not represent the full diversity of real resumes.
2. Programmatic annotations are structurally checked against exact character offsets, but have not been independently reviewed by human annotators.
3. Some fields and layouts are repeated by design. This small starter set is for learning and pipeline development, not a final claim of real-world accuracy.
4. The phone numbers are intentionally fake placeholders. They must not be dialled or treated as real contact details.
5. Training/testing on this synthetic set alone cannot establish performance on real candidate resumes. Before claiming real-world performance, evaluate on a separately collected, consented and appropriately secured test set.

## Recommended use
Use the `train.spacy` file for model training, `validation.spacy` for validation, and `test.spacy` only for final held-out evaluation. The `.spacy` files use spaCy's DocBin format. The JSONL annotation files are retained for transparency and manual inspection.

## Reproducibility
Dataset generation seed: `20261125`. The files in this archive are the generated snapshot for this project.
