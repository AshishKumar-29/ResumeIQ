# Annotation Guide

Annotate only the exact text spans that carry the target information. Do not annotate section headings (e.g., `SKILLS`, `EDUCATION`). Do not expand spans to include labels such as `Email:` or punctuation separators.

| Label | Include | Exclude |
|---|---|---|
| PERSON | Candidate's full name | Section titles, names of companies |
| EMAIL | Complete email address | `Email:` prefix |
| PHONE | Complete phone string | `Phone:` prefix |
| SKILL | One individual skill phrase | Commas/bullets separating skills |
| DEGREE | Full qualification name | Institution and graduation year |
| INSTITUTION | College, university or institute name | Degree and year |
| JOB_TITLE | Exact role/designation | Employer name |
| COMPANY | Employer/company name | Role and dates |
| DATE | Graduation year or employment date/range | Duration phrase |
| DURATION | Phrase such as `2 years` | Employment date range |

Offsets are zero-based Unicode character positions, with inclusive `start` and exclusive `end`. Always verify that `text[start:end]` exactly equals the annotated value.
