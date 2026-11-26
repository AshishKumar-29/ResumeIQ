"""Evaluate ResumeIQ NER v2 on the original held-out test set plus new hard cases.

Run from the main ResumeIQ project folder:
    python evaluate_resume_ner_v2.py

This is evaluation only; it does not retrain or modify either saved model.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import spacy
from spacy.tokens import DocBin
from spacy.training import Example

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset" / "ResumeIQ_Phase2_Dataset"
TEST_FILE = DATASET_DIR / "spacy" / "test.spacy"
AUGMENTATION_FILE = DATASET_DIR / "annotations" / "ner_augmentation_v2.jsonl"
MODEL_DIR = PROJECT_DIR / "models" / "resume_ner_v2"
REPORT_FILE = PROJECT_DIR / "outputs" / "ner_test_results_v2.json"


def examples_from_docbin(path: Path, nlp: spacy.language.Language) -> list[Example]:
    if not path.is_file():
        raise FileNotFoundError(f"Test data not found: {path}")
    docs = list(DocBin().from_disk(path).get_docs(nlp.vocab))
    if not docs:
        raise ValueError(f"No documents found in {path}")
    examples = []
    for gold in docs:
        entities = [(e.start_char, e.end_char, e.label_) for e in gold.ents]
        examples.append(Example.from_dict(nlp.make_doc(gold.text), {"entities": entities}))
    return examples


def load_augmented_test(path: Path, nlp: spacy.language.Language) -> list[Example]:
    if not path.is_file():
        raise FileNotFoundError(f"Augmentation data not found: {path}")
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != "test":
                continue
            text = record["text"]
            entities = record["entities"]
            previous_end = -1
            for start, end, label in entities:
                if start < previous_end or text[start:end] == "":
                    raise ValueError(f"Invalid annotation in {record.get('id', line_number)}")
                if nlp.make_doc(text).char_span(start, end, label=label, alignment_mode="strict") is None:
                    raise ValueError(f"Token alignment failed in {record.get('id', line_number)}")
                previous_end = end
            examples.append(Example.from_dict(nlp.make_doc(text), {"entities": entities}))
    if not examples:
        raise ValueError("No held-out augmentation examples were found.")
    return examples


def metrics(nlp: spacy.language.Language, examples: list[Example]) -> dict[str, Any]:
    scores = nlp.evaluate(examples)
    return {
        "documents": len(examples),
        "precision": float(scores.get("ents_p", 0.0)),
        "recall": float(scores.get("ents_r", 0.0)),
        "f1": float(scores.get("ents_f", 0.0)),
        "per_entity": scores.get("ents_per_type", {}),
    }


def main() -> None:
    print(f"spaCy version: {spacy.__version__}")
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Model not found: {MODEL_DIR}. Run train_resume_ner_v2.py first.")
    nlp = spacy.load(MODEL_DIR)
    original_test = examples_from_docbin(TEST_FILE, nlp)
    hard_case_test = load_augmented_test(AUGMENTATION_FILE, nlp)
    combined = original_test + hard_case_test

    original_metrics = metrics(nlp, original_test)
    hard_metrics = metrics(nlp, hard_case_test)
    combined_metrics = metrics(nlp, combined)

    for title, result in [
        ("Original held-out test set", original_metrics),
        ("New-layout hard-case test set", hard_metrics),
        ("Combined test sets", combined_metrics),
    ]:
        print(f"\n{title}")
        print(f"Documents: {result['documents']}")
        print(f"Precision: {result['precision']:.3f}")
        print(f"Recall:    {result['recall']:.3f}")
        print(f"F1-score:  {result['f1']:.3f}")

    report = {
        "model_version": "v2",
        "spacy_version": spacy.__version__,
        "original_test": original_metrics,
        "new_layout_hard_case_test": hard_metrics,
        "combined_test": combined_metrics,
        "note": (
            "The augmentation test examples are synthetic and were held out from training and validation. "
            "These scores are still not a substitute for evaluation on a larger consented real-resume dataset."
        ),
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {REPORT_FILE}")
    print("Evaluation complete. No model weights were changed.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"\nERROR: {error}") from error
