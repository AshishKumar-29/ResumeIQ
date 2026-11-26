"""Train ResumeIQ NER v2 using the original dataset plus curated synthetic examples.

Run from the main ResumeIQ project folder:
    python train_resume_ner_v2.py

This script does NOT overwrite models/resume_ner. It saves the candidate model to
models/resume_ner_v2 so the current working model remains available for rollback.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset" / "ResumeIQ_Phase2_Dataset"
SPACY_DIR = DATASET_DIR / "spacy"
AUGMENTATION_FILE = DATASET_DIR / "annotations" / "ner_augmentation_v2.jsonl"
MODEL_DIR = PROJECT_DIR / "models" / "resume_ner_v2"
REPORT_FILE = PROJECT_DIR / "outputs" / "ner_training_history_v2.json"

EPOCHS = 50
PATIENCE = 10
DROPOUT = 0.25
RANDOM_SEED = 42


def load_docbin(path: Path, vocab: Any) -> list:
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    docs = list(DocBin().from_disk(path).get_docs(vocab))
    if not docs:
        raise ValueError(f"No documents found in {path}")
    return docs


def load_augmentation(path: Path, split: str, nlp: spacy.language.Language) -> list[Example]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Augmentation file not found: {path}\n"
            "Place ner_augmentation_v2.jsonl in dataset/ResumeIQ_Phase2_Dataset/annotations/."
        )

    examples: list[Example] = []
    labels_seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if record.get("split") != split:
                continue
            text = record.get("text")
            entities = record.get("entities")
            if not isinstance(text, str) or not text.strip() or not isinstance(entities, list):
                raise ValueError(f"Malformed augmentation record on line {line_number}.")

            previous_end = -1
            converted: list[tuple[int, int, str]] = []
            doc = nlp.make_doc(text)
            for entity in entities:
                if not isinstance(entity, list) or len(entity) != 3:
                    raise ValueError(f"Bad entity in {record.get('id', line_number)}: {entity!r}")
                start, end, label = entity
                if not isinstance(start, int) or not isinstance(end, int) or not isinstance(label, str):
                    raise ValueError(f"Bad entity types in {record.get('id', line_number)}: {entity!r}")
                if start < previous_end or start < 0 or end <= start or end > len(text):
                    raise ValueError(f"Invalid or overlapping span in {record.get('id', line_number)}: {entity!r}")
                span = doc.char_span(start, end, label=label, alignment_mode="strict")
                if span is None:
                    raise ValueError(
                        f"Offset/token alignment error in {record.get('id', line_number)}: "
                        f"{text[start:end]!r} [{start}:{end}] {label}"
                    )
                converted.append((start, end, label))
                labels_seen.add(label)
                previous_end = end

            examples.append(Example.from_dict(nlp.make_doc(text), {"entities": converted}))

    if not examples:
        raise ValueError(f"No augmentation examples found for split={split!r} in {path}")
    return examples


def examples_from_docs(docs: list, nlp: spacy.language.Language) -> list[Example]:
    examples: list[Example] = []
    for gold_doc in docs:
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in gold_doc.ents]
        examples.append(Example.from_dict(nlp.make_doc(gold_doc.text), {"entities": entities}))
    return examples


def count_labels(examples: list[Example]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example in examples:
        for ent in example.reference.ents:
            counts[ent.label_] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    random.seed(RANDOM_SEED)
    print(f"spaCy version: {spacy.__version__}")
    print("Loading original DocBins and the v2 synthetic augmentation...")

    nlp = spacy.blank("en")
    base_train_docs = load_docbin(SPACY_DIR / "train.spacy", nlp.vocab)
    base_validation_docs = load_docbin(SPACY_DIR / "validation.spacy", nlp.vocab)

    train_examples = examples_from_docs(base_train_docs, nlp)
    validation_examples = examples_from_docs(base_validation_docs, nlp)
    train_examples.extend(load_augmentation(AUGMENTATION_FILE, "train", nlp))
    validation_examples.extend(load_augmentation(AUGMENTATION_FILE, "validation", nlp))

    print(f"Training documents:   {len(train_examples)} (42 original + 12 added)")
    print(f"Validation documents: {len(validation_examples)} (9 original + 4 added)")
    print(f"Training entities:    {sum(count_labels(train_examples).values())}")
    print("Training labels:")
    for label, count in count_labels(train_examples).items():
        print(f"  {label}: {count}")

    ner = nlp.add_pipe("ner")
    for label in sorted(count_labels(train_examples)):
        ner.add_label(label)

    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, float | int]] = []

    print("\nStarting v2 NER training...\n")
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train_examples)
        losses: dict[str, float] = {}
        for batch in minibatch(train_examples, size=8):
            nlp.update(batch, sgd=optimizer, drop=DROPOUT, losses=losses)

        scores = nlp.evaluate(validation_examples)
        precision = float(scores.get("ents_p", 0.0))
        recall = float(scores.get("ents_r", 0.0))
        f1 = float(scores.get("ents_f", 0.0))
        history.append({
            "epoch": epoch,
            "ner_loss": float(losses.get("ner", 0.0)),
            "validation_precision": precision,
            "validation_recall": recall,
            "validation_f1": f1,
        })
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | Loss: {losses.get('ner', 0.0):.3f} | "
            f"Val P: {precision:.3f} | Val R: {recall:.3f} | Val F1: {f1:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            no_improvement = 0
            nlp.to_disk(MODEL_DIR)
        else:
            no_improvement += 1
        if no_improvement >= PATIENCE:
            print(f"\nEarly stopping: no validation F1 improvement for {PATIENCE} epochs.")
            break

    report = {
        "model_version": "v2",
        "spacy_version": spacy.__version__,
        "base_training_documents": len(base_train_docs),
        "augmentation_training_documents": 12,
        "total_training_documents": len(train_examples),
        "base_validation_documents": len(base_validation_docs),
        "augmentation_validation_documents": 4,
        "total_validation_documents": len(validation_examples),
        "training_entity_counts": count_labels(train_examples),
        "best_epoch": best_epoch,
        "best_validation_f1": best_f1,
        "history": history,
        "note": (
            "This model uses a small synthetic dataset and a curated synthetic augmentation. "
            "Validation and test scores are project experiments, not production accuracy claims."
        ),
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTraining finished.")
    print(f"Best validation F1: {best_f1:.3f} (epoch {best_epoch})")
    print(f"Candidate model saved to: {MODEL_DIR}")
    print(f"Training report saved to: {REPORT_FILE}")
    print("The existing models/resume_ner model has not been changed.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"\nERROR: {error}") from error
