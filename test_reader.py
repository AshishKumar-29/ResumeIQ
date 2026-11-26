"""Command-line test for the ResumeIQ document reader.

Examples:
    python test_reader.py sample_resumes/RES001.pdf
    python test_reader.py sample_resumes/RES001.docx
"""

import sys
from pathlib import Path

from document_reader import extract_text


def main() -> int:
    default_file = Path("sample_resumes") / "RES001.pdf"
    file_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_file

    if not file_path.is_file():
        print(f"File not found: {file_path}")
        print("Usage: python test_reader.py <path-to-resume.pdf-or-docx>")
        return 1

    try:
        file_bytes = file_path.read_bytes()
        extracted_text = extract_text(file_bytes, file_path.name)
    except (OSError, ValueError) as exc:
        print(f"Extraction failed: {exc}")
        return 1

    print("\n========== EXTRACTED RESUME TEXT ==========\n")
    print(extracted_text)
    print("\n========== EXTRACTION SUCCESSFUL ==========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
