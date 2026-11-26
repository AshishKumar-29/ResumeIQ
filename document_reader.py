"""Read selectable text from PDF and DOCX resume files.

DOCX support recursively visits paragraphs and nested tables. This is important
for resumes that use Word tables to create multi-column layouts.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Iterator, Union

import pymupdf
from docx import Document
from docx.document import Document as DocumentObject
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P

FileInput = Union[str, Path, bytes, bytearray]

# Common template advertisements and instructions that are not resume content.
_AD_LINE_PATTERNS = (
    re.compile(r"^\s*try our resume builder\b", re.IGNORECASE),
    re.compile(r"^\s*\d+\.\s*fill in your details\b", re.IGNORECASE),
    re.compile(r"^\s*\d+\.\s*choose from \d+ designs\b", re.IGNORECASE),
    re.compile(r"^\s*\d+\.\s*download as many different design\b", re.IGNORECASE),
    re.compile(r"^\s*\$\s*[\d,.]+\s+for full access\b", re.IGNORECASE),
    re.compile(r"^\s*free\s*!?\s*$", re.IGNORECASE),
)


def _read_input(source: FileInput) -> tuple[bytes | None, Path | None]:
    """Return file bytes for byte/path inputs, preserving the path when present."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), None
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_bytes(), path


def _is_advertisement_line(line: str) -> bool:
    return any(pattern.search(line) for pattern in _AD_LINE_PATTERNS)


def _clean_docx_paragraph_text(text: str) -> str:
    """Remove template-advertisement lines while retaining resume content."""
    kept_lines: list[str] = []
    for line in text.replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if not stripped or _is_advertisement_line(stripped):
            continue
        # Remove the trailing call-to-action URL-like wording used by this template.
        if re.match(r"(?i)^try our resume builder\s*:", stripped):
            continue
        kept_lines.append(stripped)
    return "\n".join(kept_lines).strip()


def _iter_docx_blocks(parent: DocumentObject | _Cell, seen_cells: set[object]) -> Iterator[str]:
    """Yield paragraph text in document order, descending into nested tables."""
    if isinstance(parent, DocumentObject):
        container = parent.element.body
    else:
        container = parent._tc

    for child in container.iterchildren():
        if isinstance(child, CT_P):
            paragraph = Paragraph(child, parent)
            value = _clean_docx_paragraph_text(paragraph.text)
            if value:
                yield value
        elif isinstance(child, CT_Tbl):
            table = Table(child, parent)
            for row in table.rows:
                for cell in row.cells:
                    # Merged cells can appear more than once in row.cells.
                    cell_key = cell._tc
                    if cell_key in seen_cells:
                        continue
                    seen_cells.add(cell_key)
                    yield from _iter_docx_blocks(cell, seen_cells)


def _extract_docx_text(data: bytes) -> str:
    """Extract text from regular paragraphs, tables, and nested layout tables."""
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"Could not read the DOCX file: {exc}") from exc

    blocks = list(_iter_docx_blocks(document, set()))
    return "\n\n".join(blocks).strip()


def _extract_pdf_text(data: bytes) -> str:
    """Extract selectable text from each PDF page using PyMuPDF."""
    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            pages = [page.get_text("text") for page in pdf]
    except Exception as exc:
        raise ValueError(f"Could not read the PDF file: {exc}") from exc
    return "\n\n".join(page.strip() for page in pages if page.strip()).strip()


def extract_text(source: FileInput, filename: str | None = None) -> str:
    """Extract text from a PDF or DOCX.

    Args:
        source: File path or file bytes. Streamlit can pass uploaded bytes.
        filename: Required when ``source`` is bytes; used to identify file type.

    Returns:
        Extracted text with paragraph boundaries retained.
    """
    data, path = _read_input(source)
    actual_name = filename or (path.name if path else "")
    suffix = Path(actual_name).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf_text(data or b"")
    if suffix == ".docx":
        return _extract_docx_text(data or b"")
    raise ValueError("Unsupported file format. Please provide a PDF or DOCX file.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read text from a PDF or DOCX resume.")
    parser.add_argument("file", help="Path to a PDF or DOCX file")
    args = parser.parse_args()
    extracted = extract_text(args.file)
    print("\n========== EXTRACTED RESUME TEXT ==========\n")
    print(extracted or "(No selectable text found.)")
    print("\n========== EXTRACTION SUCCESSFUL ==========")
