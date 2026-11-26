"""Validate annotated resume text and convert JSONL splits to spaCy DocBin.

Important: The character offsets in the annotation JSONL files refer to the
original resume text. Do not run the text through the cleaning/preprocessing
module before converting these annotations.
"""

import json
from collections import Counter
from pathlib import Path

import spacy
from spacy.tokens import DocBin


DATASET_DIR = Path("dataset") / "ResumeIQ_Phase2_Dataset"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
SPACY_DIR = DATASET_DIR / "spacy"
SPLITS = ("train", "validation", "test")


def convert_split(split_name: str, nlp):
    """Validate one JSONL split and write its spaCy .spacy file."""
    input_path = ANNOTATIONS_DIR / f"{split_name}.jsonl"
    output_path = SPACY_DIR / f"{split_name}.spacy"

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {input_path}\n"
            "Check that the Phase 2 dataset is inside dataset/ResumeIQ_Phase2_Dataset/."
        )

    doc_bin = DocBin(store_user_data=True)
    entity_counts = Counter()
    document_count = 0

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {input_path}, line {line_number}: {error}"
                ) from error

            resume_id = record.get("id", f"line-{line_number}")
            text = record.get("text")
            entities = record.get("entities")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing or empty text for {resume_id}.")
            if not isinstance(entities, list):
                raise ValueError(f"Missing entities list for {resume_id}.")

            doc = nlp.make_doc(text)
            spans = []
            previous_end = -1

            for entity in entities:
                if not isinstance(entity, list) or len(entity) != 3:
                    raise ValueError(
                        f"Malformed entity in {resume_id}: {entity!r}. "
                        "Expected [start, end, label]."
                    )

                start, end, label = entity
                if not isinstance(start, int) or not isinstance(end, int):
                    raise ValueError(f"Non-integer offsets in {resume_id}: {entity!r}")
                if not isinstance(label, str) or not label:
                    raise ValueError(f"Invalid entity label in {resume_id}: {entity!r}")
                if start < 0 or end <= start or end > len(text):
                    raise ValueError(f"Out-of-range entity in {resume_id}: {entity!r}")
                if start < previous_end:
                    raise ValueError(
                        f"Overlapping or unsorted entities in {resume_id}. "
                        "Review the annotations before training."
                    )

                span = doc.char_span(start, end, label=label, alignment_mode="strict")
                if span is None:
                    actual = text[start:end]
                    raise ValueError(
                        f"Entity offsets do not align with spaCy tokens in {resume_id}: "
                        f"{actual!r} [{start}:{end}] label={label}."
                    )

                spans.append(span)
                entity_counts[label] += 1
                previous_end = end

            # Setting doc.ents also checks that entities do not overlap.
            doc.ents = spans
            doc_bin.add(doc)
            document_count += 1

    SPACY_DIR.mkdir(parents=True, exist_ok=True)
    doc_bin.to_disk(output_path)

    return {
        "split": split_name,
        "documents": document_count,
        "entities": sum(entity_counts.values()),
        "entity_counts": dict(sorted(entity_counts.items())),
        "output": str(output_path),
    }


def main():
    if not ANNOTATIONS_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset annotations folder not found: {ANNOTATIONS_DIR}\n"
            "Run this script from the main ResumeIQ project folder."
        )

    # A blank English pipeline provides tokenization without requiring a
    # downloaded pretrained model. The NER model will be trained later.
    nlp = spacy.blank("en")
    reports = []

    print("ResumeIQ – NLP data preparation")
    print(f"Reading annotations from: {ANNOTATIONS_DIR}\n")

    for split_name in SPLITS:
        report = convert_split(split_name, nlp)
        reports.append(report)
        print(
            f"{split_name.upper():10} | "
            f"Documents: {report['documents']:2} | "
            f"Entities: {report['entities']:3} | "
            f"Saved: {report['output']}"
        )
        print(f"Labels: {report['entity_counts']}\n")

    report_path = SPACY_DIR / "nlp_preparation_report.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(reports, report_file, indent=2, ensure_ascii=False)

    print(f"Preparation report saved to: {report_path}")
    print("All annotation offsets aligned successfully.")
    print("NLP DATA PREPARATION COMPLETE")


if __name__ == "__main__":
    main()
