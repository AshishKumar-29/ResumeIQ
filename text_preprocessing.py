"""Conservative text cleaning for ResumeIQ resume documents.

The goal is to remove formatting noise without deleting information that may
be important for information extraction (names, emails, phone numbers,
qualifications, dates, punctuation, and section headings).
"""

import re
import unicodedata


# Control characters that can appear in extracted document text.
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_WHITESPACE = re.compile(r"[\t\f\v ]+")


def preprocess_resume_text(text: str) -> str:
    """Clean extracted resume text while preserving its useful structure.

    Steps:
    1. Validate that input is text.
    2. Normalize compatible Unicode characters.
    3. Standardize line endings and remove invisible formatting characters.
    4. Collapse repeated spaces/tabs within each line.
    5. Remove empty lines at the beginning/end and repeated blank lines.

    This function intentionally does NOT lowercase text, remove punctuation,
    remove stopwords, stem words, or lemmatize words. Those operations could
    damage names, email addresses, phone numbers, dates, acronyms, and skills.
    """
    if not isinstance(text, str):
        raise TypeError("Resume text must be a string.")

    if not text.strip():
        raise ValueError("Resume text is empty. Extract text before preprocessing.")

    # Normalize visually equivalent Unicode characters (e.g. full-width forms).
    text = unicodedata.normalize("NFKC", text)

    # Normalize line endings and common invisible characters from documents.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")   # non-breaking space
    text = text.replace("\u200b", "")    # zero-width space
    text = text.replace("\ufeff", "")    # byte-order mark
    text = text.replace("\u00ad", "")    # soft hyphen
    text = _CONTROL_CHARACTERS.sub("", text)

    cleaned_lines = []

    for raw_line in text.split("\n"):
        # Collapse repeated horizontal whitespace but preserve line breaks.
        line = _HORIZONTAL_WHITESPACE.sub(" ", raw_line).strip()

        if line:
            cleaned_lines.append(line)
        elif cleaned_lines and cleaned_lines[-1] != "":
            # Keep at most one blank line between sections.
            cleaned_lines.append("")

    # Remove blank lines at the end (leading blanks never get added).
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    cleaned_text = "\n".join(cleaned_lines)

    if not cleaned_text:
        raise ValueError("No readable text remains after preprocessing.")

    return cleaned_text
