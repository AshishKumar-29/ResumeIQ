"""Test section identification using a PDF or DOCX resume."""

import argparse
from pathlib import Path

from document_reader import extract_text
from text_preprocessing import preprocess_resume_text
from section_identifier import identify_resume_sections


def main():
    parser = argparse.ArgumentParser(
        description="Extract common sections from a PDF or DOCX resume."
    )
    parser.add_argument(
        "resume",
        nargs="?",
        default="sample_resumes/RES001.pdf",
        help="Path to a PDF or DOCX resume.",
    )
    args = parser.parse_args()

    file_path = Path(args.resume)
    if not file_path.is_file():
        raise SystemExit(f"File not found: {file_path}")

    raw_text = extract_text(file_path.read_bytes(), file_path.name)
    cleaned_text = preprocess_resume_text(raw_text)
    sections = identify_resume_sections(cleaned_text)

    print("\n========== DETECTED RESUME SECTIONS ==========")
    for name, content in sections.items():
        print(f"\n[{name.upper().replace('_', ' ')}]")
        print(content if content else "(Not found)")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{file_path.stem}_sections.txt"

    with output_path.open("w", encoding="utf-8") as output_file:
        for name, content in sections.items():
            output_file.write(f"[{name.upper()}]\n")
            output_file.write(content or "(Not found)")
            output_file.write("\n\n")

    print(f"\nSection results saved to: {output_path}")
    print("========== SECTION IDENTIFICATION COMPLETE ==========")


if __name__ == "__main__":
    main()
