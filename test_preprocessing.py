"""Test ResumeIQ text preprocessing on a PDF or DOCX resume.

Examples:
    python test_preprocessing.py sample_resumes/RES001.pdf
    python test_preprocessing.py sample_resumes/RES001.docx
"""

import sys
from pathlib import Path

from document_reader import extract_text
from text_preprocessing import preprocess_resume_text


def main() -> int:
    default_file = Path("sample_resumes") / "RES001.pdf"
    file_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_file

    if not file_path.is_file():
        print(f"File not found: {file_path}")
        print("Usage: python test_preprocessing.py <path-to-resume.pdf-or-docx>")
        return 1

    try:
        file_bytes = file_path.read_bytes()
        raw_text = extract_text(file_bytes, file_path.name)
        cleaned_text = preprocess_resume_text(raw_text)
    except (OSError, TypeError, ValueError) as exc:
        print(f"Preprocessing test failed: {exc}")
        return 1

    print("\n========== ORIGINAL EXTRACTED TEXT ==========\n")
    print(raw_text)
    print("\n========== CLEANED RESUME TEXT ==========\n")
    print(cleaned_text)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{file_path.stem}_cleaned.txt"
    output_file.write_text(cleaned_text, encoding="utf-8")

    print(f"\nCleaned text saved to: {output_file}")
    print("========== PREPROCESSING SUCCESSFUL ==========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
