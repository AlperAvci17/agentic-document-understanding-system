import re
import streamlit as st
import pdfplumber
import pandas as pd
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Agentic Document Understanding System",
    page_icon="📄",
    layout="wide"
)


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("📄 Project Info")
st.sidebar.write("**Agentic Document Understanding System**")
st.sidebar.write(
    "This system analyzes uploaded documents, extracts text, generates summaries, "
    "finds keywords, classifies documents, evaluates outputs, and compares multiple documents."
)

st.sidebar.write("### Supported Files")
st.sidebar.write("- PDF")
st.sidebar.write("- TXT")
st.sidebar.write("- DOCX")

st.sidebar.write("### Main Features")
st.sidebar.write("- Text extraction")
st.sidebar.write("- Preprocessing")
st.sidebar.write("- Summarization")
st.sidebar.write("- Keyword extraction")
st.sidebar.write("- Document classification")
st.sidebar.write("- Agent decisions")
st.sidebar.write("- Rule-based evaluation")
st.sidebar.write("- Similarity analysis")
st.sidebar.write("- Downloadable reports")

st.sidebar.write("### Team Members")
st.sidebar.write("- Alper Avcı")
st.sidebar.write("- Selin Keskin")

st.sidebar.write("### Course")
st.sidebar.write("SEN4018 - Agentic AI / Data Science Project")


# -----------------------------
# MAIN TITLE
# -----------------------------

st.title("📄 Agentic Document Understanding System")

st.write(
    "Upload PDF, TXT, or DOCX documents. "
    "The system will extract text, clean it, summarize it, extract keywords, "
    "classify the document, evaluate the output, and calculate similarity "
    "between multiple documents."
)

st.info(
    "This application works as an agentic document analysis assistant. "
    "It checks the uploaded document, decides which analysis steps are needed, "
    "and generates structured outputs for the user."
)

with st.expander("ℹ️ How does the system work?"):
    st.write(
        """
        1. The user uploads one or more documents.
        2. The system extracts readable text from the files.
        3. The extracted text is cleaned and preprocessed.
        4. The agent checks document length and number of uploaded files.
        5. The system generates a summary, keywords, category, and evaluation score.
        6. If multiple documents are uploaded, similarity analysis is performed.
        7. The user can download document analysis and similarity reports.
        """
    )


# -----------------------------
# 1. TEXT EXTRACTION FUNCTIONS
# -----------------------------

def extract_text_from_pdf(file):
    text = ""

    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    except Exception as error:
        st.error(f"PDF text extraction error: {error}")
        return ""

    return text


def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8", errors="ignore")

    except Exception as error:
        st.error(f"TXT text extraction error: {error}")
        return ""


def extract_text_from_docx(file):
    text = ""

    try:
        document = Document(file)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"

    except Exception as error:
        st.error(f"DOCX text extraction error: {error}")
        return ""

    return text


def extract_text(file):
    file_name = file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(file)

    elif file_name.endswith(".txt"):
        return extract_text_from_txt(file)

    elif file_name.endswith(".docx"):
        return extract_text_from_docx(file)

    else:
        return ""


# -----------------------------
# 2. PREPROCESSING FUNCTION
# -----------------------------

def preprocess_text(text):
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


# -----------------------------
# 3. SIMPLE SUMMARIZATION
# -----------------------------

def generate_simple_summary(text, max_sentences=3):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    clean_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence.split()) > 4:
            clean_sentences.append(sentence)

    if not clean_sentences:
        return "The text is too short to generate a meaningful summary."

    summary = " ".join(clean_sentences[:max_sentences])

    return summary


# -----------------------------
# 4. KEYWORD EXTRACTION
# -----------------------------

def extract_keywords(text, top_n=8):
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=top_n
        )

        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        keyword_scores = list(zip(feature_names, scores))
        keyword_scores = sorted(keyword_scores, key=lambda x: x[1], reverse=True)

        keywords = [keyword for keyword, score in keyword_scores]

        return keywords

    except Exception:
        words = re.findall(r"\b\w+\b", text.lower())
        words = [word for word in words if len(word) > 4]

        unique_words = []

        for word in words:
            if word not in unique_words:
                unique_words.append(word)

        return unique_words[:top_n]


# -----------------------------
# 5. DOCUMENT CLASSIFICATION
# -----------------------------

def classify_document(text):
    text_lower = text.lower()

    categories = {
        "Academic Paper": [
            "abstract", "methodology", "references", "literature review",
            "experiment", "research paper", "hypothesis", "citation"
        ],
        "Lecture Note": [
            "lecture", "chapter", "course", "lesson", "definition",
            "example", "concept", "explains", "basic concepts", "notes",
            "öğrenme", "ders", "konu"
        ],
        "Report": [
            "report", "analysis", "findings", "recommendation",
            "summary", "result", "conclusion", "rapor"
        ],
        "Business Document": [
            "business", "market", "customer", "sales", "strategy",
            "company", "marketing", "revenue"
        ],
        "Technical Document": [
            "system", "software", "algorithm", "model", "database",
            "architecture", "implementation", "api", "python",
            "machine learning", "artificial intelligence", "dataset",
            "datasets", "classification", "regression", "prediction",
            "training", "testing", "evaluation", "supervised learning",
            "unsupervised learning", "data", "model evaluation"
        ],
        "News Article": [
            "news", "announced", "reported", "according to",
            "said", "breaking", "article"
        ]
    }

    scores = {}

    for category, keywords in categories.items():
        score = 0

        for keyword in keywords:
            if keyword in text_lower:
                if category == "Technical Document":
                    score += 2
                else:
                    score += 1

        scores[category] = score

    best_category = max(scores, key=scores.get)

    if scores[best_category] == 0:
        return "Other", "No strong category keywords were found."

    reason = f"The document contains keywords related to {best_category}."

    return best_category, reason


# -----------------------------
# 6. RULE-BASED EVALUATION
# -----------------------------

def evaluate_output(text, summary, keywords, category):
    score = 0
    feedback = []

    if len(text.split()) > 20:
        score += 1
        feedback.append("Text extraction is successful.")
    else:
        feedback.append("Extracted text is too short.")

    if summary and "too short" not in summary.lower():
        score += 1
        feedback.append("Summary is generated.")
    else:
        feedback.append("Summary quality is weak.")

    if len(keywords) >= 3:
        score += 1
        feedback.append("Keywords are extracted.")
    else:
        feedback.append("Not enough keywords are extracted.")

    if category != "Other":
        score += 1
        feedback.append("Document category is identified.")
    else:
        feedback.append("Document category is uncertain.")

    if len(text.split()) > 50:
        score += 1
        feedback.append("Document has enough content for analysis.")
    else:
        feedback.append("Document is short, so analysis may be limited.")

    return score, feedback


def get_quality_label(score):
    if score == 5:
        return "🟢 Excellent"
    elif score == 4:
        return "🟢 Good"
    elif score == 3:
        return "🟡 Acceptable"
    else:
        return "🔴 Needs Improvement"


# -----------------------------
# 7. SIMILARITY ANALYSIS
# -----------------------------

def interpret_similarity(score):
    if score >= 0.70:
        return "High Similarity"
    elif score >= 0.40:
        return "Medium Similarity"
    else:
        return "Low Similarity"


def calculate_similarity(documents):
    texts = [doc["clean_text"] for doc in documents]
    names = [doc["file_name"] for doc in documents]

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)

        similarity_matrix = cosine_similarity(tfidf_matrix)

        rows = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                score = round(similarity_matrix[i][j], 2)

                rows.append({
                    "Document 1": names[i],
                    "Document 2": names[j],
                    "Similarity Score": score,
                    "Interpretation": interpret_similarity(score)
                })

        return pd.DataFrame(rows)

    except Exception as error:
        st.warning(f"Similarity analysis could not be completed: {error}")
        return pd.DataFrame()


# -----------------------------
# 8. AGENT DECISION LOGIC
# -----------------------------

def agent_decision(text, number_of_files):
    decisions = []

    word_count = len(text.split())

    if word_count == 0:
        decisions.append("No readable text found. The system should warn the user.")

    elif word_count < 20:
        decisions.append("The document is very short. The system can analyze it, but results may be limited.")

    elif word_count > 800:
        decisions.append("The document is long. The system should use chunking before deeper analysis.")

    else:
        decisions.append("The document length is suitable for direct analysis.")

    if number_of_files > 1:
        decisions.append("Multiple documents uploaded. The system should calculate document similarity.")
    else:
        decisions.append("Single document uploaded. The system will skip similarity analysis.")

    return decisions


# -----------------------------
# 9. REPORT GENERATION
# -----------------------------

def generate_analysis_report(
    file_name,
    clean_text,
    summary,
    keywords,
    category,
    category_reason,
    evaluation_score,
    evaluation_feedback,
    decisions
):
    quality_label = get_quality_label(evaluation_score)

    report = f"""
DOCUMENT ANALYSIS REPORT
========================

File Name:
{file_name}

Basic Document Information:
- Text Length: {len(clean_text)} characters
- Word Count: {len(clean_text.split())} words

Document Category:
- Category: {category}
- Reason: {category_reason}

Agent Decisions:
"""

    for decision in decisions:
        report += f"- {decision}\n"

    report += f"""

Extracted Keywords:
{", ".join(keywords)}

Generated Summary:
{summary}

Evaluation:
- Score: {evaluation_score}/5
- Quality Label: {quality_label}

Evaluation Feedback:
"""

    for feedback in evaluation_feedback:
        report += f"- {feedback}\n"

    report += """

End of Report
========================
"""

    return report


def generate_similarity_report(similarity_df):
    report = """
SIMILARITY ANALYSIS REPORT
==========================

"""

    if similarity_df.empty:
        report += "No similarity result was generated.\n"

    else:
        for _, row in similarity_df.iterrows():
            report += f"Document 1: {row['Document 1']}\n"
            report += f"Document 2: {row['Document 2']}\n"
            report += f"Similarity Score: {row['Similarity Score']}\n"
            report += f"Interpretation: {row['Interpretation']}\n"
            report += "--------------------------\n"

    report += """

End of Similarity Report
========================
"""

    return report


# -----------------------------
# 10. STREAMLIT INTERFACE
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

processed_documents = []

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")
    st.header("📌 Analysis Results")

    for index, uploaded_file in enumerate(uploaded_files):
        st.divider()
        st.subheader(f"📄 File: {uploaded_file.name}")

        extracted_text = extract_text(uploaded_file)
        clean_text = preprocess_text(extracted_text)

        if clean_text:
            summary = generate_simple_summary(clean_text)
            keywords = extract_keywords(clean_text)
            category, category_reason = classify_document(clean_text)

            evaluation_score, evaluation_feedback = evaluate_output(
                clean_text,
                summary,
                keywords,
                category
            )

            decisions = agent_decision(clean_text, len(uploaded_files))
            quality_label = get_quality_label(evaluation_score)

            processed_documents.append({
                "file_name": uploaded_file.name,
                "clean_text": clean_text,
                "summary": summary,
                "keywords": keywords,
                "category": category,
                "category_reason": category_reason,
                "evaluation_score": evaluation_score,
                "evaluation_feedback": evaluation_feedback,
                "decisions": decisions
            })

            col1, col2 = st.columns(2)

            with col1:
                st.write("### Basic Document Info")
                st.write(f"**File name:** {uploaded_file.name}")
                st.write(f"**Text length:** {len(clean_text)} characters")
                st.write(f"**Word count:** {len(clean_text.split())} words")

                st.write("### Agent Decisions")
                for decision in decisions:
                    st.write(f"- {decision}")

                st.write("### Document Category")
                st.write(f"**Category:** {category}")
                st.write(f"**Reason:** {category_reason}")

            with col2:
                st.write("### Extracted Keywords")
                st.write(", ".join(keywords))

                st.write("### Evaluation Score")
                st.metric(
                    label="Output Quality",
                    value=f"{evaluation_score}/5",
                    delta=quality_label
                )

                for item in evaluation_feedback:
                    st.write(f"- {item}")

            st.write("### Generated Summary")
            st.info(summary)

            report_text = generate_analysis_report(
                uploaded_file.name,
                clean_text,
                summary,
                keywords,
                category,
                category_reason,
                evaluation_score,
                evaluation_feedback,
                decisions
            )

            st.download_button(
                label="📥 Download Analysis Report",
                data=report_text,
                file_name=f"{uploaded_file.name}_analysis_report.txt",
                mime="text/plain",
                key=f"download_report_{index}_{uploaded_file.name}"
            )

            st.write("### Extracted Text Preview")
            st.text_area(
                "First 1500 characters",
                clean_text[:1500],
                height=250,
                key=f"preview_{index}_{uploaded_file.name}"
            )

        else:
            st.warning("No readable text could be extracted from this file.")

    if len(processed_documents) > 1:
        st.divider()
        st.header("📊 Similarity Analysis Between Documents")

        similarity_df = calculate_similarity(processed_documents)

        if not similarity_df.empty:
            st.dataframe(similarity_df)

            similarity_report = generate_similarity_report(similarity_df)

            st.download_button(
                label="📥 Download Similarity Report",
                data=similarity_report,
                file_name="similarity_analysis_report.txt",
                mime="text/plain",
                key="download_similarity_report"
            )

        else:
            st.warning("Similarity analysis could not generate a result.")

else:
    st.info("Please upload at least one document.")