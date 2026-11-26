"""ResumeIQ Streamlit dashboard (polished UI, NER v2.1 pipeline).

Run from the project root:
    python -m streamlit run app.py
"""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

# Use the latest tested hybrid extractor and the separate NER v2 model.
from resume_extractor_v2_1 import extract_resume_data, load_resume_model

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

st.set_page_config(
    page_title="ResumeIQ | Smart Resume Parser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
      --ri-bg: #0b1220;
      --ri-panel: #111c2e;
      --ri-panel-2: #16243a;
      --ri-line: rgba(148, 163, 184, .18);
      --ri-muted: #9aaac0;
      --ri-text: #e8eef7;
      --ri-accent: #60a5fa;
      --ri-accent-2: #38bdf8;
      --ri-green: #34d399;
    }
    [data-testid="stAppViewContainer"] {
      background: radial-gradient(circle at 80% 0%, rgba(37,99,235,.12), transparent 30%), var(--ri-bg);
      color: var(--ri-text);
    }
    [data-testid="stHeader"] { background: rgba(11,18,32,.78); }
    .main .block-container {
      max-width: 1240px;
      padding-top: 2.2rem;
      padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
      background: #0e1929;
      border-right: 1px solid var(--ri-line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #c5d1e0; }
    .ri-brand-row { display:flex; align-items:center; gap:12px; margin-bottom: 4px; }
    .ri-logo {
      width:44px; height:44px; display:flex; align-items:center; justify-content:center;
      border-radius:13px; font-size:23px; background:linear-gradient(135deg,#2563eb,#0ea5e9);
      box-shadow:0 8px 24px rgba(37,99,235,.28);
    }
    .ri-brand { font-size:2rem; font-weight:800; letter-spacing:-.04em; color:#f8fbff; }
    .ri-kicker { color:#7dd3fc; text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; margin-top:1.2rem; }
    .ri-subtitle { color:#aebed2; font-size:1rem; max-width:780px; line-height:1.6; margin-top:.25rem; }
    .ri-upload-panel {
      background:linear-gradient(135deg,rgba(30,58,95,.48),rgba(17,28,46,.94));
      border:1px solid rgba(96,165,250,.25); border-radius:18px; padding:1.15rem 1.25rem .65rem;
      margin:1.4rem 0 1rem;
    }
    .ri-panel-title { font-size:1rem; font-weight:700; color:#eaf3ff; margin-bottom:.15rem; }
    .ri-panel-note { color:var(--ri-muted); font-size:.86rem; margin-bottom:.6rem; }
    div[data-testid="stFileUploader"] section {
      background:rgba(8,17,31,.42); border:1px dashed rgba(125,211,252,.42); border-radius:13px;
    }
    div[data-testid="stFileUploader"] section:hover { border-color:#60a5fa; }
    .stButton > button[kind="primary"] {
      border:0; border-radius:11px; min-height:2.85rem; font-weight:700;
      background:linear-gradient(90deg,#2563eb,#0284c7); color:white;
      box-shadow:0 8px 22px rgba(37,99,235,.22);
    }
    .stButton > button[kind="primary"]:hover { border:0; filter:brightness(1.08); }
    .ri-file-chip {
      display:inline-flex; align-items:center; gap:8px; padding:7px 11px; border-radius:999px;
      background:rgba(96,165,250,.12); border:1px solid rgba(96,165,250,.25); color:#bfdbfe;
      font-size:.83rem; margin:.4rem 0 .7rem;
    }
    .ri-section-heading { font-size:1.18rem; font-weight:750; letter-spacing:-.02em; margin:.35rem 0 .85rem; color:#eff6ff; }
    .ri-contact-card {
      background:linear-gradient(145deg,rgba(22,36,58,.92),rgba(17,28,46,.92));
      border:1px solid var(--ri-line); border-radius:14px; padding:14px 16px; min-height:94px;
    }
    .ri-contact-label { color:#91a4bd; text-transform:uppercase; letter-spacing:.1em; font-size:.68rem; font-weight:750; margin-bottom:8px; }
    .ri-contact-value { color:#f3f7fc; font-size:1.03rem; font-weight:650; overflow-wrap:anywhere; }
    .ri-stat-card {
      background:rgba(17,28,46,.9); border:1px solid var(--ri-line); border-radius:13px;
      padding:13px 15px; min-height:83px;
    }
    .ri-stat-label { color:#91a4bd; font-size:.75rem; font-weight:650; }
    .ri-stat-value { color:#f8fbff; font-size:1.65rem; font-weight:800; line-height:1.25; margin-top:5px; }
    .ri-list-item {
      padding:9px 12px; border:1px solid rgba(148,163,184,.14); border-radius:10px;
      background:rgba(17,28,46,.58); margin-bottom:7px; color:#dce7f5;
    }
    .ri-empty { color:#93a4ba; background:rgba(148,163,184,.07); border:1px dashed rgba(148,163,184,.22); border-radius:10px; padding:12px; }
    .ri-summary {
      background:rgba(14,165,233,.08); border-left:3px solid #38bdf8;
      padding:14px 16px; border-radius:0 11px 11px 0; color:#dceafa; line-height:1.65;
    }
    .ri-footer { color:#8293aa; font-size:.78rem; border-top:1px solid var(--ri-line); padding-top:1rem; margin-top:2.2rem; }
    [data-testid="stTabs"] button { font-weight:650; }
    [data-testid="stExpander"] { border:1px solid var(--ri-line); border-radius:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the ResumeIQ NER v2 model...")
def get_nlp_model():
    return load_resume_model()


def _list_text(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return "; ".join(str(value) for value in values if str(value).strip())


def make_csv(result: dict[str, Any]) -> str:
    education = result.get("education", {}) or {}
    experience = result.get("experience", {}) or {}
    row = {
        "file_name": result.get("file_name", ""),
        "name": result.get("name", ""),
        "email": result.get("email", ""),
        "phone": result.get("phone", ""),
        "skills": _list_text(result.get("skills", [])),
        "degrees": _list_text(education.get("degrees", [])),
        "other_qualifications": _list_text(education.get("qualifications", [])),
        "institutions": _list_text(education.get("institutions", [])),
        "job_titles": _list_text(experience.get("job_titles", [])),
        "companies": _list_text(experience.get("companies", [])),
        "dates": _list_text(experience.get("dates", [])),
        "durations": _list_text(experience.get("durations", [])),
        "professional_summary": result.get("professional_summary", ""),
    }
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(row.keys()))
    writer.writeheader()
    writer.writerow(row)
    return buffer.getvalue()


def save_json(result: dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(result.get("file_name", "resume")).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "resume"
    output_path = OUTPUT_DIR / f"{safe_stem}_v2_1_extracted.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _values(values: Any) -> list[str]:
    if isinstance(values, list):
        return [str(v).strip() for v in values if str(v).strip()]
    return []


def render_list(values: Any, empty_text: str = "No information detected") -> None:
    items = _values(values)
    if not items:
        st.markdown(f'<div class="ri-empty">{html.escape(empty_text)}</div>', unsafe_allow_html=True)
        return
    for item in items:
        st.markdown(f'<div class="ri-list-item">{html.escape(item)}</div>', unsafe_allow_html=True)


def render_contact(label: str, value: Any) -> None:
    display = str(value).strip() if value else "Not found"
    st.markdown(
        '<div class="ri-contact-card">'
        f'<div class="ri-contact-label">{html.escape(label)}</div>'
        f'<div class="ri-contact-value">{html.escape(display)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: int) -> None:
    st.markdown(
        '<div class="ri-stat-card">'
        f'<div class="ri-stat-label">{html.escape(label)}</div>'
        f'<div class="ri-stat-value">{value}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def show_results(result: dict[str, Any], output_path: Path) -> None:
    education = result.get("education", {}) or {}
    experience = result.get("experience", {}) or {}
    skills = _values(result.get("skills", []))
    degrees = _values(education.get("degrees", []))
    qualifications = _values(education.get("qualifications", []))
    institutions = _values(education.get("institutions", []))
    job_titles = _values(experience.get("job_titles", []))
    companies = _values(experience.get("companies", []))
    dates = _values(experience.get("dates", []))
    durations = _values(experience.get("durations", []))
    summary = str(result.get("professional_summary", "") or "").strip()

    st.markdown('<div class="ri-section-heading">Extracted contact details</div>', unsafe_allow_html=True)
    contact_cols = st.columns(3)
    with contact_cols[0]:
        render_contact("Full name", result.get("name"))
    with contact_cols[1]:
        render_contact("Email address", result.get("email"))
    with contact_cols[2]:
        render_contact("Phone number", result.get("phone"))

    st.markdown("<br>", unsafe_allow_html=True)
    stat_cols = st.columns(4)
    with stat_cols[0]:
        stat_card("SKILLS FOUND", len(skills))
    with stat_cols[1]:
        stat_card("EDUCATION ITEMS", len(degrees) + len(qualifications) + len(institutions))
    with stat_cols[2]:
        stat_card("EXPERIENCE ITEMS", len(job_titles) + len(companies))
    with stat_cols[3]:
        stat_card("NER ENTITIES", len(result.get("entities", []) or []))

    st.markdown("<br>", unsafe_allow_html=True)
    overview_tab, skills_tab, education_tab, experience_tab = st.tabs(
        ["Overview", "Skills", "Education", "Experience"]
    )

    with overview_tab:
        st.markdown('<div class="ri-section-heading">Professional summary</div>', unsafe_allow_html=True)
        if summary:
            st.markdown(f'<div class="ri-summary">{html.escape(summary)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ri-empty">No professional summary detected.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top extracted skills**")
            render_list(skills[:8], "No skills detected")
        with c2:
            st.markdown("**Experience snapshot**")
            render_list(job_titles, "No job titles detected")
            render_list(companies, "No companies detected")

    with skills_tab:
        st.markdown('<div class="ri-section-heading">Detected skills</div>', unsafe_allow_html=True)
        if skills:
            # Compact chips keep long skill lists readable.
            chips = " ".join(
                f'<span style="display:inline-block;margin:4px 5px 4px 0;padding:7px 11px;'
                f'border:1px solid rgba(96,165,250,.28);border-radius:999px;'
                f'background:rgba(37,99,235,.12);color:#dbeafe;font-size:.88rem;">{html.escape(x)}</span>'
                for x in skills
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown('<div class="ri-empty">No skills detected in this resume.</div>', unsafe_allow_html=True)

    with education_tab:
        st.markdown('<div class="ri-section-heading">Education details</div>', unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Degrees**")
            render_list(degrees)
            st.markdown("**Other qualifications**")
            render_list(qualifications)
        with e2:
            st.markdown("**Institutions**")
            render_list(institutions)

    with experience_tab:
        st.markdown('<div class="ri-section-heading">Work experience</div>', unsafe_allow_html=True)
        x1, x2 = st.columns(2)
        with x1:
            st.markdown("**Job titles**")
            render_list(job_titles)
            st.markdown("**Companies**")
            render_list(companies)
        with x2:
            st.markdown("**Employment dates**")
            render_list(dates)
            st.markdown("**Durations**")
            render_list(durations)

    with st.expander("Advanced diagnostics · raw NER predictions"):
        st.caption("These are the model's raw entity predictions. Some may be corrected or filtered by the hybrid extraction rules above.")
        entities = result.get("entities", []) or []
        if entities:
            st.dataframe(entities, use_container_width=True, hide_index=True)
        else:
            st.caption("No named entities were detected.")

    st.markdown('<div class="ri-section-heading">Export your extracted information</div>', unsafe_allow_html=True)
    json_bytes = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
    csv_bytes = make_csv(result).encode("utf-8-sig")
    file_stem = Path(result.get("file_name", "resume")).stem
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "⬇  Download JSON", data=json_bytes,
            file_name=f"{file_stem}_v2_1_extracted.json", mime="application/json",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "⬇  Download CSV", data=csv_bytes,
            file_name=f"{file_stem}_v2_1_extracted.csv", mime="text/csv",
            use_container_width=True,
        )
    st.caption(f"A JSON copy is also saved locally in: `{output_path}`")
    st.info("ResumeIQ is a student capstone prototype. Please review extracted information before using it.")


# -----------------------------------------------------------------------------
# Page layout
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📄 ResumeIQ")
    st.caption("AI-assisted resume information extraction")
    st.divider()
    st.markdown("**Supported documents**")
    st.write("PDF · DOCX")
    st.markdown("**Extraction pipeline**")
    st.write("Regex + custom spaCy NER v2 + section-aware rules")
    st.markdown("**Export formats**")
    st.write("JSON · CSV")
    st.divider()
    st.markdown("**How it works**")
    st.caption("1. Upload a resume\n\n2. Extract its text\n\n3. Identify key information\n\n4. Review and export results")
    st.divider()
    st.caption("Scanned/image-only PDFs are not supported in this version.")

st.markdown(
    '<div class="ri-brand-row"><div class="ri-logo">📄</div><div class="ri-brand">ResumeIQ</div></div>'
    '<div class="ri-kicker">Resume intelligence workspace</div>'
    '<div class="ri-subtitle">Turn resumes into structured information. Upload a PDF or Word document to extract contact details, skills, education and work experience.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="ri-upload-panel">'
    '<div class="ri-panel-title">Upload your resume</div>'
    '<div class="ri-panel-note">Choose a text-based PDF or DOCX file. Maximum recommended size: 10 MB.</div>'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a PDF or DOCX file",
    type=["pdf", "docx"],
    accept_multiple_files=False,
    help="Upload a PDF with selectable text or a Microsoft Word DOCX document.",
    label_visibility="visible",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    upload_id = hashlib.sha256(file_bytes).hexdigest()
    extension = Path(uploaded_file.name).suffix.lower()

    if extension not in {".pdf", ".docx"}:
        st.error("Unsupported file format. Please upload a PDF or DOCX resume.")
    elif len(file_bytes) == 0:
        st.error("The uploaded file is empty. Please choose another file.")
    elif len(file_bytes) > 10 * 1024 * 1024:
        st.error("This file is larger than 10 MB. Please upload a smaller resume.")
    else:
        if st.session_state.get("resume_upload_id") != upload_id:
            st.session_state.pop("resume_result", None)
            st.session_state.pop("resume_output_path", None)
            st.session_state["resume_upload_id"] = upload_id

        st.markdown(
            f'<div class="ri-file-chip">✓ &nbsp; {html.escape(uploaded_file.name)} &nbsp;·&nbsp; {len(file_bytes) / 1024:.1f} KB</div>',
            unsafe_allow_html=True,
        )
        if st.button("✨  Analyse resume", type="primary", use_container_width=True):
            try:
                with st.spinner("Reading the document and extracting resume information..."):
                    nlp = get_nlp_model()
                    result = extract_resume_data(file_bytes, uploaded_file.name, nlp=nlp)
                    output_path = save_json(result)
                st.session_state["resume_result"] = result
                st.session_state["resume_output_path"] = str(output_path)
                st.success("Analysis complete. Review the extracted information below.")
            except Exception as exc:
                st.error(f"Could not process this resume: {exc}")
                st.caption("Check that the trained NER v2 model and required project files are present, and that the PDF contains selectable text.")

        saved_result = st.session_state.get("resume_result")
        saved_path = st.session_state.get("resume_output_path")
        if saved_result and saved_path:
            st.divider()
            show_results(saved_result, Path(saved_path))
else:
    st.session_state.pop("resume_result", None)
    st.session_state.pop("resume_output_path", None)
    st.session_state.pop("resume_upload_id", None)
    st.markdown(
        '<div class="ri-empty">No resume uploaded yet. Select a PDF or DOCX file above to get started.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="ri-footer">ResumeIQ · Student capstone prototype · Hybrid NLP information extraction · Review results before making decisions.</div>',
    unsafe_allow_html=True,
)
