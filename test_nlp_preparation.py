"""Check the spaCy training files created by prepare_nlp_data.py."""

from collections import Counter
from pathlib import Path

import spacy
from spacy.tokens import DocBin


SPACY_DIR = Path("dataset") / "ResumeIQ_Phase2_Dataset" / "spacy"
EXPECTED_DOCUMENTS = {"train": 42, "validation": 9, "test": 9}


def main():
    if not SPACY_DIR.is_dir():
        raise SystemExit(
            f"Could not find {SPACY_DIR}. Run prepare_nlp_data.py first."
        )

    nlp = spacy.blank("en")
    all_passed = True

    print("ResumeIQ – checking prepared spaCy datasets\n")

    for split_name, expected_count in EXPECTED_DOCUMENTS.items():
        file_path = SPACY_DIR / f"{split_name}.spacy"
        if not file_path.is_file():
            print(f"[FAIL] Missing file: {file_path}")
            all_passed = False
            continue

        doc_bin = DocBin().from_disk(file_path)
        docs = list(doc_bin.get_docs(nlp.vocab))
        label_counts = Counter(
            entity.label_ for doc in docs for entity in doc.ents
        )
        entity_total = sum(label_counts.values())

        count_ok = len(docs) == expected_count
        entities_ok = entity_total > 0
        status = "PASS" if count_ok and entities_ok else "FAIL"
        print(
            f"[{status}] {split_name}: {len(docs)} documents, "
            f"{entity_total} entities"
        )
        print(f"       Labels: {dict(sorted(label_counts.items()))}")

        if not (count_ok and entities_ok):
            all_passed = False

    if not all_passed:
        raise SystemExit("One or more NLP preparation checks failed.")

    print("\nAll NLP preparation checks passed.")


if __name__ == "__main__":
    main()
