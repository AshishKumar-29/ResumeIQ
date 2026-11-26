"""Identify common sections in a resume using heading-based rules.

This is a first-pass section detector. It identifies section boundaries from
common headings; it does not yet extract individual skills, degrees, or jobs.
"""

import re
from typing import Dict, List, Optional, Tuple


SECTION_ALIASES = {
    "professional_summary": {
        "summary", "professional summary", "profile", "professional profile",
        "career objective", "objective", "about me", "personal summary",
    },
    "skills": {
        "skills", "technical skills", "key skills", "core skills",
        "technical expertise", "areas of expertise", "competencies",
        "skills and competencies",
    },
    "education": {
        "education", "educational background", "academic background",
        "academic qualifications", "qualifications",
    },
    "experience": {
        "experience", "work experience", "professional experience",
        "employment history", "work history", "career history",
    },
    "projects": {
        "projects", "academic projects", "personal projects",
        "project experience",
    },
    "certifications": {
        "certifications", "certificates", "licenses and certifications",
        "professional certifications",
    },
}

def _normalise_heading(line: str) -> str:
    """Normalise a candidate heading for matching."""
    value = line.strip().lower()
    value = re.sub(r"^[\W_]+|[\W_]+$", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def _build_heading_lookup() -> Dict[str, str]:
    lookup = {}
    for section_name, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            lookup[alias] = section_name
    return lookup


_HEADING_LOOKUP = _build_heading_lookup()


def identify_resume_sections(text: str) -> Dict[str, str]:
    """Return detected resume sections and their text content.

    Only standalone heading lines are treated as section headings. The content
    after a heading continues until the next recognised heading. Missing
    sections are returned as empty strings.
    """
    if not isinstance(text, str):
        raise TypeError("Resume text must be a string.")
    if not text.strip():
        raise ValueError("Resume text is empty.")

    sections = {name: "" for name in SECTION_ALIASES}
    current_section: Optional[str] = None
    collected: Dict[str, List[str]] = {name: [] for name in SECTION_ALIASES}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_section and collected[current_section]:
                collected[current_section].append("")
            continue

        heading = _normalise_heading(line)
        detected = _HEADING_LOOKUP.get(heading)

        if detected:
            current_section = detected
            continue

        if current_section:
            collected[current_section].append(line)

    for section_name, lines in collected.items():
        # Keep at most one blank line between content lines.
        content = "\n".join(lines).strip()
        content = re.sub(r"\n{3,}", "\n\n", content)
        sections[section_name] = content

    return sections


def find_detected_sections(text: str) -> Dict[str, str]:
    """Return only sections that contain extracted content."""
    all_sections = identify_resume_sections(text)
    return {name: content for name, content in all_sections.items() if content}
