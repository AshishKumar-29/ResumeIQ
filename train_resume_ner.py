"""Train a custom spaCy NER model for the ResumeIQ project.

Expected project layout:
    dataset/ResumeIQ_Phase2_Dataset/spacy/train.spacy
    dataset/ResumeIQ_Phase2_Dataset/spacy/validation.spacy

Run from the project root:
    python train_resume_ner.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import spacy
from spacy.training import Example
from spacy.util import minibatch
from spacy.tokens import DocBin


PROJECT_DIR = Path(__file__).resolve().parent
SPACY_DATA_DIR = PROJECT_DIR / "dataset" / "ResumeIQ_Phase2_Dataset" / "spacy"
TRAIN_FILE = SPACY_DATA_DIR / "train.spacy"
VALIDATION_FILE = SPACY_DATA_DIR / "validation.spacy"
MODEL_DIR = PROJECT_DIR / "models" / "resume_ner"
REPORT_FILE = PROJECT_DIR / "outputs" / "ner_training_history.json"

# This dataset is intentionally small and synthetic. These settings are a
# sensible starting point for the project, not a guarantee of production accuracy.
EPOCHS = 40
PATIENCE = 8
DROPOUT = 0.20
RANDOM_SEED = 42


def load_docs(file_path: Path, vocab: Any) -> list:
    """Load annotated spaCy Docs from a .spacy DocBin file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Check that the dataset folders are in the project root."
        )

    doc_bin = DocBin().from_disk(file_path)
    docs = list(doc_bin.get_docs(vocab))
    if not docs:
        raise ValueError(f"No documents were found in {file_path}.")
    return docs


def make_examples(docs: list, nlp: spacy.language.Language) -> list[Example]:
    """Convert annotated Docs into spaCy training Examples."""
    examples: list[Example] = []
    for gold_doc in docs:
        entities = [
            (ent.start_char, ent.end_char, ent.label_)
            for ent in gold_doc.ents
        ]
        example = Example.from_dict(
            nlp.make_doc(gold_doc.text),
            {"entities": entities},
        )
        examples.append(example)
    return examples


def get_entity_counts(docs: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for doc in docs:
        for ent in doc.ents:
            counts[ent.label_] = counts.get(ent.label_, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    random.seed(RANDOM_SEED)
    print(f"spaCy version: {spacy.__version__}")
    print("Loading the annotated training and validation data...")

    # Load the DocBins using a temporary blank English vocabulary.
    nlp = spacy.blank("en")
    train_docs = load_docs(TRAIN_FILE, nlp.vocab)
    validation_docs = load_docs(VALIDATION_FILE, nlp.vocab)

    print(f"Training documents:   {len(train_docs)}")
    print(f"Validation documents: {len(validation_docs)}")
    print(f"Training entities:    {sum(get_entity_counts(train_docs).values())}")
    print("Entity labels:")
    for label, count in get_entity_counts(train_docs).items():
        print(f"  {label}: {count}")

    # Add one NER component and register every label found in the training set.
    ner = nlp.add_pipe("ner")
    labels = sorted({ent.label_ for doc in train_docs for ent in doc.ents})
    if not labels:
        raise ValueError("The training data contains no entity labels.")
    for label in labels:
        ner.add_label(label)

    train_examples = make_examples(train_docs, nlp)

    # spaCy initializes its model parameters using the examples supplied here.
    optimizer = nlp.initialize(get_examples=lambda: train_examples)

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    print("\nStarting NER training...\n")
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train_examples)
        losses: dict[str, float] = {}

        batches = minibatch(train_examples, size=8)
        for batch in batches:
            nlp.update(
                batch,
                sgd=optimizer,
                drop=DROPOUT,
                losses=losses,
            )

        # Create fresh validation Examples for each evaluation pass.
        validation_examples = make_examples(validation_docs, nlp)
        scores = nlp.evaluate(validation_examples)
        precision = float(scores.get("ents_p", 0.0))
        recall = float(scores.get("ents_r", 0.0))
        f1 = float(scores.get("ents_f", 0.0))
        ner_loss = float(losses.get("ner", 0.0))

        history.append(
            {
                "epoch": epoch,
                "ner_loss": ner_loss,
                "validation_precision": precision,
                "validation_recall": recall,
                "validation_f1": f1,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"NER loss: {ner_loss:.3f} | "
            f"Val P: {precision:.3f} | "
            f"Val R: {recall:.3f} | "
            f"Val F1: {f1:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            epochs_without_improvement = 0
            nlp.to_disk(MODEL_DIR)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping: validation F1 did not improve for {PATIENCE} epochs.")
            break

    report = {
        "spacy_version": spacy.__version__,
        "training_documents": len(train_docs),
        "validation_documents": len(validation_docs),
        "training_entity_counts": get_entity_counts(train_docs),
        "best_epoch": best_epoch,
        "best_validation_f1": best_f1,
        "history": history,
        "note": (
            "Metrics are from a small synthetic dataset and should be treated as project-level "
            "experimental results, not evidence of production performance."
        ),
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTraining finished.")
    print(f"Best validation F1: {best_f1:.3f} (epoch {best_epoch})")
    print(f"Best model saved to: {MODEL_DIR}")
    print(f"Training report saved to: {REPORT_FILE}")
    print("\nNext, we will test the saved model on the separate test.spacy dataset.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"\nERROR: {error}") from error
