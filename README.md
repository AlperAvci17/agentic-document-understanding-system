# Agentic Document Understanding System

## Project Overview

Agentic Document Understanding System is an AI-based document analysis application developed for the SEN4018 course project.

The system allows users to upload PDF, TXT, or DOCX documents and automatically analyzes their content. It extracts readable text, preprocesses the document, generates a summary, extracts important keywords, classifies the document, evaluates the output quality, and compares multiple documents using similarity analysis.

This project follows an agentic AI approach because the system can make decisions during the analysis process. For example, it checks whether the document is readable, whether the text is long or short, whether multiple documents are uploaded, and whether similarity analysis should be performed.

---

## Main Features

- Upload PDF, TXT, and DOCX documents
- Extract text from uploaded files
- Clean and preprocess extracted text
- Generate a basic document summary
- Extract important keywords using TF-IDF
- Classify documents into categories
- Make agentic decisions based on document conditions
- Evaluate output quality with a rule-based evaluation framework
- Calculate similarity between multiple uploaded documents
- Generate downloadable document analysis reports
- Generate downloadable similarity analysis reports
- Display results through a Streamlit web interface

---

## Project Motivation

Reading and organizing long documents manually can be time-consuming. Students, researchers, and professionals often need to quickly understand the content of many documents such as reports, lecture notes, academic papers, or technical files.

This project aims to reduce this workload by creating an intelligent document understanding system. Instead of manually reading each document, users can upload their files and receive structured outputs such as summaries, keywords, categories, evaluation scores, and similarity results.

---

## Agentic AI Structure

The system includes an autonomous decision-making process.

The agent checks:

- Whether readable text can be extracted
- Whether the document is too short
- Whether the document is too long
- Whether one or multiple documents are uploaded
- Whether similarity analysis is required
- Whether the output quality is sufficient

Based on these conditions, the system decides which actions should be performed.

Example decisions:

- If the document is very short, the system warns that the analysis may be limited.
- If the document is long, the system suggests chunking for deeper analysis.
- If multiple documents are uploaded, the system performs similarity analysis.
- If only one document is uploaded, the system skips similarity analysis.

---

## System Architecture

The project consists of the following modules:

1. User Interface Module
2. Document Upload Module
3. Text Extraction Module
4. Preprocessing Module
5. Summarization Module
6. Keyword Extraction Module
7. Document Classification Module
8. Agent Decision Module
9. Rule-Based Evaluation Module
10. Similarity Analysis Module
11. Report Generation Module

---

## Technologies Used

- Python
- Streamlit
- pdfplumber
- python-docx
- pandas
- scikit-learn
- TF-IDF
- Cosine Similarity
- GitHub
- Streamlit Cloud

---

## Supported File Types

The system supports the following document types:

- PDF
- TXT
- DOCX

---

## Document Categories

The system can classify uploaded documents into categories such as:

- Academic Paper
- Lecture Note
- Report
- Business Document
- Technical Document
- News Article
- Other

---

## Evaluation Framework

The project includes a rule-based evaluation framework.

The system evaluates each document output based on:

- Text extraction success
- Summary generation
- Keyword extraction
- Document classification
- Document length and content quality

Each document receives an evaluation score out of 5.

---

## Similarity Analysis

When multiple documents are uploaded, the system calculates similarity between documents using:

- TF-IDF vectorization
- Cosine similarity

The result shows how similar two uploaded documents are.

---

## Report Generation

The system can generate downloadable reports.

There are two report types:

1. Document Analysis Report  
   Includes document name, word count, category, agent decisions, keywords, summary, evaluation score, and feedback.

2. Similarity Analysis Report  
   Includes document pairs and similarity scores.

---

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/selinkeskinn/agentic-document-understanding-system.git