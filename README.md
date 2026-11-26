# ResumeIQ – Phase 3 Starter

ResumeIQ is a capstone project for **Resume Information Extraction**. The current Phase 3 starter implements the first backend component: reading selectable text from PDF and DOCX resumes.

## Current features

- Extracts text from PDF files using PyMuPDF.
- Extracts paragraphs and tables from DOCX files using python-docx.
- Selects the reader using the file extension.
- Reports unsupported files, empty files, unreadable documents, and PDFs with no selectable text.
- Includes sample PDF and DOCX resumes from the synthetic Phase 2 dataset.

**Not implemented yet:** OCR for scanned PDFs, text preprocessing, NLP/NER extraction, Streamlit GUI, CSV/JSON export, and model training. These will be developed in later steps.

## Project structure

```text
ResumeIQ_Phase3_Starter/
├── document_reader.py
├── test_reader.py
├── requirements.txt
├── README.md
├── sample_resumes/
│   ├── RES001.pdf
│   └── RES001.docx
├── dataset/                  # Full Phase 2 dataset is included here
└── outputs/                  # Reserved for later output files
```

The complete Phase 2 dataset is included in `dataset/ResumeIQ_Phase2_Dataset/`, including synthetic resume text, annotations, train/validation/test splits, and sample upload documents.

## Setup on Windows (PowerShell)

Open the project folder in VS Code, then open its terminal:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python` is not recognised, install Python 3 and enable **Add Python to PATH** during installation, then reopen VS Code.

## Test PDF extraction

```powershell
python test_reader.py sample_resumes/RES001.pdf
```

## Test DOCX extraction

```powershell
python test_reader.py sample_resumes/RES001.docx
```

You can also pass the path to another PDF or DOCX file:

```powershell
python test_reader.py "C:\path\to\your\resume.pdf"
```

## How the reader works

1. `test_reader.py` reads the selected file as bytes.
2. It calls `extract_text(file_bytes, filename)` from `document_reader.py`.
3. `extract_text` checks the file extension.
4. PDF files are handled by `extract_pdf_text`; DOCX files are handled by `extract_docx_text`.
5. The extracted text is printed in the terminal.

## Known limitations

- Image-only/scanned PDFs do not contain selectable text. OCR support is not part of this starter yet.
- Text extraction from complex layouts, headers, footers, columns, and text boxes may be incomplete.
- This module only reads resume text. It does not yet identify skills, education, experience, names, emails, or phone numbers.

## Dataset note

The included resumes are synthetic examples created for development and testing. They do not represent real candidate records. Some contact details are deliberately non-dialable placeholders.

## Next development step

Implement `text_preprocessing.py` to normalise extracted text while preserving information needed for resume parsing (such as email addresses, phone numbers, dates, punctuation, and skill names).
