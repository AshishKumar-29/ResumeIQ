# ResumeIQ
### Resume Information Extraction using NLP and Custom spaCy NER

ResumeIQ is a resume-parsing capstone project that converts information in PDF and DOCX resumes into structured candidate details. It combines document reading, text preprocessing, section identification, a custom spaCy Named Entity Recognition (NER) model, and rule-based post-processing in a Streamlit interface.

## Features

- Upload resumes in PDF and DOCX format.
- Extract selectable text from PDFs and paragraphs/tables from DOCX files.
- Preprocess extracted text while preserving information useful for resume parsing.
- Identify common resume sections: Summary, Skills, Education, Experience, Projects, and Certifications.
- Extract names, email addresses, phone numbers, skills, degrees, institutions, job titles, companies, dates, and experience details.
- View extracted information in separate sections in the Streamlit app.
- Download extracted information as CSV or JSON.
- Use a custom-trained spaCy NER model with additional post-processing.

## Technology Stack

- Python
- Streamlit
- spaCy
- PyMuPDF
- python-docx
- Regular Expressions
- Pandas

## Project Structure

```text
ResumeIQ/
├── app.py
├── document_reader.py
├── text_preprocessing.py
├── section_identifier.py
├── resume_extractor.py
├── resume_extractor_v2.py
├── resume_extractor_v2_1.py
├── train_resume_ner.py
├── train_resume_ner_v2.py
├── evaluate_resume_ner.py
├── evaluate_resume_ner_v2.py
├── prepare_nlp_data.py
├── compare_ner_versions.py
├── test_reader.py
├── test_preprocessing.py
├── test_section_identifier.py
├── test_nlp_preparation.py
├── requirements.txt
├── README.md
├── dataset/
├── models/
│   ├── resume_ner/
│   └── resume_ner_v2/
├── sample_resumes/
├── outputs/
└── Presentation/
```

## Installation and Setup (Windows)

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

5. Run the application:

```powershell
python -m streamlit run app.py
```

Open the local address shown in the terminal. Keep the `models/` directory structure intact because the trained model files are included in the repository.

## Running the Tests

Run these commands from the project root:

```powershell
python test_reader.py
python test_preprocessing.py
python test_section_identifier.py
python test_nlp_preparation.py
```

These scripts test document reading, preprocessing, section identification, and NLP dataset preparation.

## Dataset and Model

The project uses a small, programmatically generated synthetic resume dataset for development and evaluation. It includes annotated examples and train, validation, and test splits. Synthetic resumes are also included for application testing.

The custom spaCy NER model was trained to identify these resume-related entity types:

- `PERSON`
- `EMAIL`
- `PHONE`
- `SKILL`
- `DEGREE`
- `INSTITUTION`
- `JOB_TITLE`
- `COMPANY`
- `DATE`
- `DURATION`

Rule-based and post-processing logic is also used to improve extraction of structured resume fields.

**Privacy Note:** Synthetic examples are for development and testing. Do not upload or commit private resumes or personal candidate information without permission. The real resume used during local testing is not included in this repository.

## Model Evaluation

The v2 NER model was evaluated on synthetic held-out data:

| Evaluation Set | Documents | Precision / Recall / F1 |
|---|---:|---:|
| Original held-out test set | 9 | 0.993 |
| New-layout hard-case test set | 4 | 0.911 |
| Combined test set | 13 | 0.971 |

These figures measure entity extraction on the project's small synthetic test sets. They are not a guarantee of performance on arbitrary real-world resumes. The dataset is limited in size and layout diversity, and broader testing with independently annotated resumes is needed.

## Limitations

- Image-only or scanned PDFs may not yield text because OCR is not implemented.
- Complex layouts, columns, graphics, and unusual formatting can affect extraction.
- Resume formats and wording vary, so some fields may be missed or incorrectly classified.
- The evaluation dataset is small and synthetic; real-world generalisation has not been established.

## Team Members

| Name | Roll Number |
|---|---|
| Ashish Kumar | 2401201215 |
| Garima Sharma | 2401201212 |
| Vikram | 2401201217 |
| Krish | 2401201218 |

## Project Presentation

The project presentation is available in the `Presentation/` folder.
