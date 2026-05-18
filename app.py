import re
import streamlit as st
import pdfplumber
import pandas as pd
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Agentic Document Understanding System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Agentic Document Understanding System")

st.write(
    "Upload PDF, TXT, or DOCX documents. "
    "The system will extract text, clean it, summarize it, extract keywords, "
    "classify the document, and evaluate the output."
)


# -----------------------------
# 1. TEXT EXTRACTION FUNCTIONS
# -----------------------------

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_txt(file):
    return file.read().decode("utf-8", errors="ignore")


def extract_text_from_docx(file):
    document = Document(file)
    text = ""
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"
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
        keywords = vectorizer.get_feature_names_out()
        return list(keywords)

    except:
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


# -----------------------------
# 7. SIMILARITY ANALYSIS
# -----------------------------

def calculate_similarity(documents):
    texts = [doc["clean_text"] for doc in documents]
    names = [doc["file_name"] for doc in documents]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rows.append({
                "Document 1": names[i],
                "Document 2": names[j],
                "Similarity Score": round(similarity_matrix[i][j], 2)
            })

    return pd.DataFrame(rows)


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
# 9. STREAMLIT INTERFACE
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

processed_documents = []

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")

    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(f"📌 File: {uploaded_file.name}")

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

            processed_documents.append({
                "file_name": uploaded_file.name,
                "clean_text": clean_text,
                "summary": summary,
                "keywords": keywords,
                "category": category,
                "evaluation_score": evaluation_score
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
                st.write(f"**Score:** {evaluation_score}/5")

                for item in evaluation_feedback:
                    st.write(f"- {item}")

            st.write("### Generated Summary")
            st.info(summary)

            st.write("### Extracted Text Preview")
            st.text_area(
                "First 1500 characters",
                clean_text[:1500],
                height=250,
                key=uploaded_file.name
            )

        else:
            st.warning("No readable text could be extracted from this file.")

    if len(processed_documents) > 1:
        st.divider()
        st.header("📊 Similarity Analysis Between Documents")

        similarity_df = calculate_similarity(processed_documents)
        st.dataframe(similarity_df)

else:
    st.info("Please upload at least one document.")
