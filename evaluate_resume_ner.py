"""Evaluate the saved ResumeIQ spaCy NER model on the untouched test set.

Run from the project root:
    python evaluate_resume_ner.py

This script does not train or modify the saved model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import spacy
from spacy.tokens import DocBin
from spacy.training import Example


PROJECT_DIR = Path(__file__).resolve().parent
TEST_FILE = PROJECT_DIR / "dataset" / "ResumeIQ_Phase2_Dataset" / "spacy" / "test.spacy"
MODEL_DIR = PROJECT_DIR / "models" / "resume_ner"
REPORT_FILE = PROJECT_DIR / "outputs" / "ner_test_results.json"


def load_test_docs(file_path: Path, vocab: Any) -> list:
    """Load annotated test documents from a spaCy DocBin file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Check the dataset folder and project root."
        )

    docs = list(DocBin().from_disk(file_path).get_docs(vocab))
    if not docs:
        raise ValueError(f"No test documents were found in {file_path}.")
    return docs


def make_examples(gold_docs: list, nlp: spacy.language.Language) -> list[Example]:
    """Create spaCy evaluation Examples with the original gold annotations."""
    examples: list[Example] = []
    for gold_doc in gold_docs:
        entities = [
            (ent.start_char, ent.end_char, ent.label_)
            for ent in gold_doc.ents
        ]
        examples.append(
            Example.from_dict(
                nlp.make_doc(gold_doc.text),
                {"entities": entities},
            )
        )
    return examples


def count_entities(docs: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for doc in docs:
        for ent in doc.ents:
            counts[ent.label_] = counts.get(ent.label_, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    print(f"spaCy version: {spacy.__version__}")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Saved model not found at {MODEL_DIR}. Run python train_resume_ner.py first."
        )

    print(f"Loading saved model from: {MODEL_DIR}")
    nlp = spacy.load(MODEL_DIR)

    print("Loading the separate test dataset...")
    test_docs = load_test_docs(TEST_FILE, nlp.vocab)
    gold_counts = count_entities(test_docs)
    examples = make_examples(test_docs, nlp)

    # Evaluation only: nlp.evaluate makes predictions but does not update weights.
    scores = nlp.evaluate(examples)
    precision = float(scores.get("ents_p", 0.0))
    recall = float(scores.get("ents_r", 0.0))
    f1 = float(scores.get("ents_f", 0.0))
    per_type = scores.get("ents_per_type", {})

    print(f"\nTest documents: {len(test_docs)}")
    print(f"Gold test entities: {sum(gold_counts.values())}")
    print("\nOverall test results (exact entity span and label match):")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    print("\nPer-entity results:")
    for label in sorted(gold_counts):
        metrics = per_type.get(label, {})
        p = float(metrics.get("p", 0.0))
        r = float(metrics.get("r", 0.0))
        entity_f1 = float(metrics.get("f", 0.0))
        print(
            f"{label:12s} | Gold: {gold_counts[label]:3d} | "
            f"P: {p:.3f} | R: {r:.3f} | F1: {entity_f1:.3f}"
        )

    report = {
        "spacy_version": spacy.__version__,
        "model_path": str(MODEL_DIR),
        "test_documents": len(test_docs),
        "gold_test_entity_counts": gold_counts,
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "per_entity": per_type,
        "note": (
            "Evaluation uses exact entity span and label matching on the separate synthetic test set. "
            "Results should not be interpreted as production performance on real-world resumes."
        ),
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nTest report saved to: {REPORT_FILE}")
    print("\nEvaluation complete. The saved model was not retrained or modified.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"\nERROR: {error}") from error
