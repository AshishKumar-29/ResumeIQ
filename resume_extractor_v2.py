"""Hybrid end-to-end information extraction for ResumeIQ.

This version combines regex, explicit line patterns, the custom spaCy NER model,
and context checks to reduce false positives in education and experience fields.

Run from the project root:
    python resume_extractor.py sample_resumes/RES001.pdf
    python resume_extractor.py sample_resumes/RES001.docx

For Streamlit:
    from resume_extractor import extract_resume_data, load_resume_model
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import spacy

from document_reader import extract_text
from text_preprocessing import preprocess_resume_text
from section_identifier import identify_resume_sections


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models" / "resume_ner_v2"
OUTPUT_DIR = PROJECT_DIR / "outputs"

EMAIL_PATTERN = re.compile(
    r"(?i)(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])"
)
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]*)?"
    r"(?:\(\s*\d{2,4}\s*\)|\d{2,4})"
    r"(?:[\s.-]*\d{2,4}){1,3}(?!\w)"
)

KNOWN_SKILLS = [
    "Object-Oriented Programming", "Unsupervised Learning", "Supervised Learning",
    "Natural Language Processing", "Machine Learning", "Deep Learning",
    "Data Structures", "API Integration", "REST APIs", "REST API",
    "Advanced Excel", "Microsoft Excel", "Power BI", "scikit-learn",
    "TensorFlow", "JavaScript", "TypeScript", "PostgreSQL", "MongoDB",
    "BeautifulSoup", "Streamlit", "Bootstrap", "Node.js", "Express.js",
    "MySQL", "SQLite", "PyTorch", "Databricks", "Salesforce", "GitHub",
    "Figma", "Django", "Flask", "React", "NLTK", "spaCy", "Python",
    "Java", "C++", "C#", "HTML", "CSS", "SQL", "Tableau", "Apex",
    "CRM", "Git", "Excel", "Prototyping", "UI Design", "Data Analysis",
    "Algorithms", "OOP", "NLP", "PHP", "R", "MS Office",
    # Common skills found in the classic two-column resume template.
    "Adobe Photoshop", "Photoshop", "Adobe Illustrator", "Illustrator",
    "After Effects", "Public Speaking", "Public Speaker", "Graphic Design",
]

SKILL_CATEGORY_LINE = re.compile(
    r"^\s*(?:programming languages|web development|database|databases|"
    r"data\s*&\s*analytics|data and analytics|design\s*&\s*ui|"
    r"tools\s*&\s*platforms|tools and platforms|ai\s*/\s*machine learning|"
    r"ai/machine learning|core concepts)\s*:\s*(.*?)\s*$",
    re.IGNORECASE,
)

# Only accept education credentials when the line begins with a known
# credential pattern. This avoids treating NER fragments such as "BCA
# Fundamentals" or trailing course names as degrees.
DEGREE_LINE = re.compile(
    r"^\s*(?:"
    r"(?:bachelor|master|associate)\b.*|"
    r"(?:BCA|MCA|MBA|B\.?\s?Tech|B\.?\s?Sc|M\.?\s?Sc|Ph\.?\s?D)"
    r"(?:\s*\(|\s*[-–—:]|\s*$).*|"
    r"higher\s+secondary\b.*|diploma\b.*"
    r")",
    re.IGNORECASE,
)
EDUCATION_LEVEL_LINE = re.compile(r"^\s*(?:10th|12th)\s+grade\b.*", re.IGNORECASE)
INSTITUTION_EXCLUSIONS = re.compile(r"\b(?:project|major project|team project)\b", re.IGNORECASE)
INSTITUTION_WORD = re.compile(
    r"(?:school|college|university|institute|institution|academy)",
    re.IGNORECASE,
)

JOB_TITLE_WORD = re.compile(
    r"\b(?:intern|developer|engineer|analyst|manager|designer|architect|"
    r"consultant|specialist|associate|administrator|executive|coordinator|"
    r"scientist|programmer|technician|lead|officer|strategist|researcher|"
    r"trainee|cashier|assistant|laborer|laboratory assistant)\b",
    re.IGNORECASE,
)
JOB_TITLE_EXCLUSIONS = re.compile(
    r"\b(?:catalyst|certificat(?:e|ion)|course|program|programme|"
    r"fundamentals|learning|basics?)\b",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b"
    r"|\b(?:19|20)\d{2}\s*(?:[-–—]|to)\s*(?:(?:19|20)\d{2}|present)\b",
    re.IGNORECASE,
)
DURATION_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:years?|yrs?|months?|mos?)\b", re.IGNORECASE
)

SKIP_NAME_LINES = {
    "resume", "curriculum vitae", "cv", "profile", "summary", "about me",
    "professional summary", "professional profile", "objective", "contact",
    "contact information", "personal details", "projects", "education",
    "technical skills", "skills", "work experience", "experience",
    "certifications", "certificates", "languages", "achievements",
}


def load_resume_model():
    """Load the trained ResumeIQ spaCy model."""
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_DIR}\n"
            "Run 'python train_resume_ner_v2.py' before extracting resume data."
        )
    try:
        return spacy.load(MODEL_DIR)
    except Exception as exc:
        raise RuntimeError(f"Could not load the ResumeIQ NER model: {exc}") from exc


def _get_entities(text: str, nlp: Any) -> list[dict[str, Any]]:
    """Run NER and return JSON-serialisable entity information."""
    if not text.strip():
        return []
    doc = nlp(text)
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        }
        for ent in doc.ents
    ]


def _unique(values: list[str]) -> list[str]:
    """Remove empty and duplicate strings while preserving order."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = re.sub(r"\s+", " ", value).strip(" \t\r\n,;|•-–—")
        key = value.casefold()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _extract_email(text: str) -> str:
    match = EMAIL_PATTERN.search(text)
    return match.group(0).strip(".,;:|)>") if match else ""


def _extract_phone(text: str) -> str:
    candidates: list[str] = []
    for match in PHONE_PATTERN.finditer(text):
        raw = match.group(0).strip(" .-|")
        digits = re.sub(r"\D", "", raw)
        if 10 <= len(digits) <= 13:
            candidates.append(raw)
    if not candidates:
        return ""
    for candidate in candidates:
        if candidate.startswith("+"):
            return candidate
    return candidates[0]


def _extract_name(text: str, entities: list[dict[str, Any]]) -> str:
    """Prefer a plausible person name in the first lines of the document."""
    for raw_line in text.splitlines()[:12]:
        line = raw_line.strip().lstrip("•*- ").strip()
        if not line or line.casefold() in SKIP_NAME_LINES:
            continue
        if any(token in line for token in ("@", "|", "http://", "https://", "www.")):
            continue
        if re.search(r"\d", line) or ":" in line:
            continue
        words = line.split()
        if not 2 <= len(words) <= 5:
            continue
        if not re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ.'’\- ]+", line):
            continue
        lowered = line.casefold()
        if any(word in lowered for word in (
            "project", "intern", "developer", "engineer", "analyst", "manager",
            "website", "university", "college", "school", "technical skills",
        )):
            continue
        return line.strip(" .")

    for entity in entities:
        if entity["label"] == "PERSON":
            return entity["text"].strip()
    return ""


def _known_skill_matches(value: str) -> list[str]:
    """Find known skill phrases, preferring longer matches over nested ones."""
    matches: list[tuple[int, int, str]] = []
    for skill in KNOWN_SKILLS:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(skill)}(?![A-Za-z0-9])", re.IGNORECASE
        )
        for match in pattern.finditer(value):
            matches.append((match.start(), match.end(), skill))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    accepted: list[tuple[int, int, str]] = []
    for start, end, skill in matches:
        if any(start < old_end and end > old_start for old_start, old_end, _ in accepted):
            continue
        accepted.append((start, end, skill))
    accepted.sort(key=lambda item: item[0])
    return [skill for _, _, skill in accepted]


def _extract_skills(text: str, entities: list[dict[str, Any]]) -> list[str]:
    """Extract skills from labelled lines, explicit skill sections, and NER.

    Some two-column templates use prose under a heading such as
    ``PROFESSIONAL SKILLS`` rather than a category line like ``Skills: Python``.
    Read that block until the next major section heading.
    """
    skills: list[str] = []
    lines = [line.strip() for line in text.splitlines()]

    # Existing structured/category-style skills (e.g. "Programming Languages: Python").
    for line in lines:
        match = SKILL_CATEGORY_LINE.match(line)
        if match:
            skills.extend(_known_skill_matches(match.group(1)))

    # Find skill headings, including templates that split the heading over
    # two separate lines: "PROFESSIONAL" followed by "SKILLS".
    skill_sections: list[str] = []
    start_indices: list[int] = []
    for index, line in enumerate(lines):
        normalized = re.sub(r"[^a-z]+", " ", line.casefold()).strip()
        if normalized in {"professional skills", "skills", "key skills", "core skills"}:
            start_indices.append(index + 1)
        elif normalized == "professional" and index + 1 < len(lines):
            next_line = re.sub(r"[^a-z]+", " ", lines[index + 1].casefold()).strip()
            if next_line == "skills":
                start_indices.append(index + 2)

    end_headings = {
        "languages", "education", "work experience", "experience",
        "references", "certifications", "certificates", "projects",
        "achievements", "professional summary", "summary", "profile",
    }
    for start in start_indices:
        block: list[str] = []
        for line in lines[start:]:
            normalized = re.sub(r"[^a-z]+", " ", line.casefold()).strip()
            if normalized in end_headings:
                break
            if line:
                block.append(line)
        if block:
            skill_sections.append(" ".join(block))

    for block in skill_sections:
        skills.extend(_known_skill_matches(block))

    # Retain model suggestions as a fallback, but canonicalise only known skills.
    if not skills:
        for entity in entities:
            if entity["label"] == "SKILL":
                skills.extend(_known_skill_matches(entity["text"]))

    cleaned = _unique(skills)
    # Merge common wording variants and avoid listing the generic and branded
    # names for the same software skill twice.
    normalized = {item.casefold() for item in cleaned}
    if "public speaker" in normalized:
        cleaned = [item for item in cleaned if item.casefold() != "public speaker"]
        cleaned.append("Public Speaking")
    if "adobe photoshop" in normalized:
        cleaned = [item for item in cleaned if item.casefold() != "photoshop"]
    if "adobe illustrator" in normalized:
        cleaned = [item for item in cleaned if item.casefold() != "illustrator"]
    return _unique(cleaned)


def _extract_education(text: str, entities: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Extract education only from explicit credential and institution lines.

    NER output is intentionally not merged here: the current small synthetic
    model over-predicts education entities on unrelated real-resume text.
    """
    degrees: list[str] = []
    qualifications: list[str] = []
    institutions: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("•*- ").strip()
        if not line:
            continue
        if DEGREE_LINE.match(line):
            degree = re.split(r"\s*[|]\s*", line, maxsplit=1)[0].strip()
            # Avoid treating unrelated lines beginning with short acronyms as degrees.
            if len(degree) >= 5:
                degrees.append(degree)
        if EDUCATION_LEVEL_LINE.match(line):
            qualifications.append(line)
        if INSTITUTION_WORD.search(line) and not INSTITUTION_EXCLUSIONS.search(line):
            candidate = re.split(r"\s*[|]\s*", line, maxsplit=1)[0].strip()
            candidate = candidate.split(",", maxsplit=1)[0].strip()
            if candidate and len(candidate) <= 100:
                institutions.append(candidate)

    return {
        "degrees": _unique(degrees),
        "qualifications": _unique(qualifications),
        "institutions": _unique(institutions),
    }


def _clean_company_candidate(line: str) -> str:
    """Return a plausible employer fragment from a title's following line."""
    value = line.strip().lstrip("•*- ").strip()
    if not value or len(value) > 90 or value.casefold() in SKIP_NAME_LINES:
        return ""
    if any(token in value.casefold() for token in (
        "github.com", "linkedin.com", "http://", "https://", "@", "|", ":"
    )):
        return ""
    if value.startswith(("Worked ", "Collaborated ", "Contributed ", "Built ", "Developed ")):
        return ""
    if INSTITUTION_WORD.search(value):
        return ""
    # Employer lines commonly include a city after a comma.
    candidate = value.split(",", maxsplit=1)[0].strip()
    if not candidate or len(candidate.split()) > 5:
        return ""
    if JOB_TITLE_WORD.search(candidate) or JOB_TITLE_EXCLUSIONS.search(candidate):
        return ""
    if not re.fullmatch(r"[A-Za-z0-9&.'’()\-/ ]+", candidate):
        return ""
    return candidate


def _extract_experience(text: str, entities: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Extract job entries from explicit work-history areas.

    Handles both:
      * title on one line, employer/location/date on the next line
      * title and employer together, separated by a comma

    Restricting title search to a detected Work Experience section prevents
    profile subtitles such as "Customer Care & Cashier" from becoming jobs.
    """
    all_lines = [line.strip() for line in text.splitlines()]
    lines = all_lines

    # Prefer lines inside an explicit experience section when that section
    # contains at least one recognisable job-title candidate. Otherwise retain
    # the broader fallback for layouts whose PDF/DOCX text order is unusual.
    heading_start = next(
        (i for i, line in enumerate(all_lines)
         if re.sub(r"[^a-z]+", " ", line.casefold()).strip()
         in {"work experience", "professional experience", "employment history", "experience"}),
        None,
    )
    if heading_start is not None:
        end_headings = {"education", "academic qualifications", "certifications", "certificates", "languages", "references"}
        heading_end = next(
            (i for i in range(heading_start + 1, len(all_lines))
             if re.sub(r"[^a-z]+", " ", all_lines[i].casefold()).strip() in end_headings),
            len(all_lines),
        )
        section_lines = all_lines[heading_start + 1:heading_end]
        if any(JOB_TITLE_WORD.search(line) and not JOB_TITLE_EXCLUSIONS.search(line)
               for line in section_lines):
            lines = section_lines

    title_indices: list[int] = []
    job_titles: list[str] = []
    inline_companies: dict[int, str] = {}

    for index, line in enumerate(lines):
        if not line:
            continue
        # "Golf Course" may be part of an employer name, not a training course.
        if JOB_TITLE_EXCLUSIONS.search(line) and not re.search(r"\bgolf\s+course\b", line, re.I):
            continue
        if not JOB_TITLE_WORD.search(line) or len(line) > 110:
            continue
        if line.endswith((".", "!", "?")):
            continue
        if re.match(r"(?i)^(?:seeking|passionate|proficient|third year|experienced|motivated)\b", line):
            continue

        candidate = re.split(r"\s*[|–—]\s*", line, maxsplit=1)[0].strip()
        inline_company = ""

        # Example: "Customer service cashier, SCOTTSDALE Golf course".
        # Split only when the left-hand side itself looks like a role.
        if "," in candidate:
            left, right = candidate.split(",", maxsplit=1)
            right_candidate = re.sub(r"\s+", " ", right).strip(" ,;|—–-")
            # Employer names such as "Scottsdale Golf Course" can contain
            # generic words (e.g. "course") that are excluded by the
            # next-line company heuristic; allow them in this title/employer
            # heading pattern.
            right_is_plausible = (
                right_candidate
                and len(right_candidate) <= 90
                and len(right_candidate.split()) <= 8
                and not any(token in right_candidate.casefold() for token in (
                    "github.com", "linkedin.com", "http://", "https://", "@"
                ))
                and not JOB_TITLE_WORD.search(right_candidate)
                and not re.search(r"\b(?:certificate|certification|program|programme|catalyst)\b", right_candidate, re.I)
            )
            if JOB_TITLE_WORD.search(left) and right_is_plausible:
                candidate = left.strip()
                inline_company = right_candidate

        if candidate and JOB_TITLE_WORD.search(candidate):
            title_indices.append(index)
            job_titles.append(candidate)
            if inline_company:
                inline_companies[index] = inline_company

    # NER suggestions are accepted only when their containing line is also
    # recognised as a role line by the context rules above.
    for entity in entities:
        if entity["label"] != "JOB_TITLE":
            continue
        ent_text = entity["text"].strip()
        if not JOB_TITLE_WORD.search(ent_text) or JOB_TITLE_EXCLUSIONS.search(ent_text):
            continue
        containing_index = next(
            (i for i, line in enumerate(lines) if ent_text.casefold() in line.casefold()),
            None,
        )
        if containing_index is not None and containing_index not in title_indices:
            line = lines[containing_index]
            if JOB_TITLE_WORD.search(line) and not JOB_TITLE_EXCLUSIONS.search(line):
                title_indices.append(containing_index)
                job_titles.append(ent_text)

    title_indices = sorted(set(title_indices))
    companies: list[str] = []
    dates: list[str] = []
    durations: list[str] = []

    for index in title_indices:
        inline_company = inline_companies.get(index, "")
        if inline_company:
            companies.append(inline_company)

        nearby = [line for line in lines[index:index + 4] if line]
        if not inline_company and len(nearby) >= 2:
            company = _clean_company_candidate(nearby[1])
            if company:
                companies.append(company)

        local_text = "\n".join(nearby)
        dates.extend(match.group(0) for match in DATE_PATTERN.finditer(local_text))
        durations.extend(match.group(0) for match in DURATION_PATTERN.finditer(local_text))

    return {
        "job_titles": _unique(job_titles),
        "companies": _unique(companies),
        "dates": _unique(dates),
        "durations": _unique(durations),
    }


def _extract_professional_summary(text: str, sections: dict[str, Any]) -> str:
    """Find a likely profile paragraph, including resumes with unusual PDF reading order.

    First use a detected PROFILE/SUMMARY section. If the heading order in a
    multi-column PDF causes the section detector to miss its paragraph, fall
    back to paragraph-level cues while excluding bullets, links and headings.
    """
    detected = sections.get("professional_summary", "")
    if isinstance(detected, str) and len(detected.strip()) >= 40:
        return re.sub(r"\s+", " ", detected).strip()

    lines = [line.strip() for line in text.splitlines()]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        # Keep wrapped paragraph lines together, but split at bullets and URLs.
        if line.startswith(("•", "- ", "* ")) or "github.com/" in line.casefold() or "linkedin.com/" in line.casefold():
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if line.upper() in {"PROFILE", "SUMMARY", "PROFESSIONAL SUMMARY", "OBJECTIVE", "ABOUT ME"}:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))

    cue_pattern = re.compile(
        r"\b(?:student|experience|proficient|passionate|seeking|adaptable|"
        r"professional|skilled|currently pursuing|career|foundation|"
        r"hands-on|looking for|opportunity|contribute|self-motivated)\b",
        re.IGNORECASE,
    )
    candidates: list[tuple[int, str]] = []
    for paragraph in paragraphs:
        candidate = re.sub(r"\s+", " ", paragraph).strip()
        if not 35 <= len(candidate) <= 900:
            continue
        if "@" in candidate or re.search(r"https?://|www\.|github\.com|linkedin\.com", candidate, re.I):
            continue
        if re.match(r"(?i)^(?:programming languages|web development|database|tools|"
                    r"technical skills|education|experience|projects|certificates)\b", candidate):
            continue
        cues = len(cue_pattern.findall(candidate))
        # Prefer a coherent profile-style paragraph over a long descriptive list.
        if cues:
            score = cues * 10 + min(len(candidate), 500) / 100
            if candidate.count(";") > 2 or candidate.count("|") > 0:
                score -= 10
            candidates.append((int(score), candidate))
    return max(candidates, key=lambda item: item[0])[1] if candidates else ""


def extract_resume_data(file_bytes: bytes, filename: str, nlp: Any | None = None) -> dict[str, Any]:
    """Extract structured information from an uploaded PDF or DOCX resume."""
    if not file_bytes:
        raise ValueError("The uploaded resume is empty.")
    model = nlp if nlp is not None else load_resume_model()
    raw_text = extract_text(file_bytes, filename)
    cleaned_text = preprocess_resume_text(raw_text)
    sections = identify_resume_sections(cleaned_text)
    all_entities = _get_entities(cleaned_text, model)

    return {
        "file_name": Path(filename).name,
        "name": _extract_name(cleaned_text, all_entities),
        "email": _extract_email(cleaned_text),
        "phone": _extract_phone(cleaned_text),
        "skills": _extract_skills(cleaned_text, all_entities),
        "education": _extract_education(cleaned_text, all_entities),
        "experience": _extract_experience(cleaned_text, all_entities),
        "professional_summary": _extract_professional_summary(cleaned_text, sections),
        "sections": sections,
        "entities": all_entities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured information from a ResumeIQ PDF or DOCX resume."
    )
    parser.add_argument("resume", help="Path to a PDF or DOCX resume")
    args = parser.parse_args()
    resume_path = Path(args.resume)
    if not resume_path.is_file():
        raise SystemExit(f"Resume file not found: {resume_path}")

    try:
        model = load_resume_model()
        result = extract_resume_data(resume_path.read_bytes(), resume_path.name, model)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"\nERROR: {exc}") from exc

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{resume_path.stem}_v2_extracted.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n========== RESUME INFORMATION ==========")
    print(f"Name:  {result['name'] or '(Not found)'}")
    print(f"Email: {result['email'] or '(Not found)'}")
    print(f"Phone: {result['phone'] or '(Not found)'}")
    print("\nSkills:")
    print(", ".join(result["skills"]) if result["skills"] else "(Not found)")
    print("\nEducation:")
    print("Degrees: " + (", ".join(result["education"]["degrees"]) or "(Not found)"))
    print("Other qualifications: " + (", ".join(result["education"]["qualifications"]) or "(Not found)"))
    print("Institutions: " + (", ".join(result["education"]["institutions"]) or "(Not found)"))
    print("\nExperience:")
    for field in ("job_titles", "companies", "dates", "durations"):
        values = result["experience"][field]
        print(f"{field.replace('_', ' ').title()}: {', '.join(values) if values else '(Not found)'}")
    print(f"\nStructured JSON saved to: {output_path}")
    print("========== EXTRACTION COMPLETE ==========")


if __name__ == "__main__":
    main()
