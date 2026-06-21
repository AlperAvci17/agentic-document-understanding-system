import re
import streamlit as st
import pdfplumber
import pandas as pd
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


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


# -----------------------------
# CUSTOM CSS - WIDE SUMMARY BOXES
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 88% !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }

    div[data-testid="stAlert"] {
        width: min(78vw, 1050px) !important;
        max-width: 78vw !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
st.sidebar.write("- LLM-based evaluation")
st.sidebar.write("- Similarity analysis")
st.sidebar.write("- Downloadable reports")

st.sidebar.write("### Team Members")
st.sidebar.write("- Alper Avcı")
st.sidebar.write("- Selin Keskin")

st.sidebar.write("### Course")
st.sidebar.write("SEN4018 - Agentic AI / Data Science Project")

st.sidebar.divider()

st.sidebar.write("### LLM Evaluation")
st.sidebar.write(
    "Enter an OpenAI API key if you want the system to perform LLM-based evaluation. "
    "If you leave it empty, the app will still work with rule-based evaluation."
)

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="This key is used only during this session and is not stored in the code."
)

llm_model = "gpt-4o-mini"
st.sidebar.text_input(
    "LLM Model",
    value=llm_model,
    disabled=True,
    help="This model is fixed and read-only."
)



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
        5. The system generates a summary, keywords, category, and rule-based evaluation score.
        6. If an API key is provided, the system also performs LLM-based evaluation.
        7. If multiple documents are uploaded, similarity analysis is performed.
        8. The user can download document analysis and similarity reports.
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
# 7. LLM-BASED EVALUATION
# -----------------------------
def generate_llm_summary(api_key, model_name, document_text):
    """
    Generate an LLM-based summary of the document.
    If API key is not available or an error occurs, return None and a status message.
    """

    if not api_key:
        return None, "LLM summary skipped because no API key was provided."

    if OpenAI is None:
        return None, "LLM summary could not run because the openai package is not installed."

    try:
        client = OpenAI(api_key=api_key)

        limited_text = document_text[:3500]

        prompt = f"""
You are a document summarization assistant.

Summarize the following document clearly and concisely.

The summary should:
- Capture the main ideas of the document
- Include important concepts
- Avoid unnecessary details
- Be written in one clear paragraph
- Be understandable for a student

Document Text:
{limited_text}
"""

        response = client.responses.create(
            model=model_name,
            input=prompt
        )

        return response.output_text, "LLM summary was generated."

    except Exception as error:
        return None, f"LLM summary could not be completed. Error: {error}"
        
def generate_llm_evaluation(
    api_key,
    model_name,
    document_text,
    summary,
    keywords,
    category
):
    if not api_key:
        return "LLM evaluation was skipped because no API key was provided."

    if OpenAI is None:
        return "LLM evaluation could not run because the openai package is not installed."

    try:
        client = OpenAI(api_key=api_key)

        limited_text = document_text[:3500]

        prompt = f"""
You are evaluating the output of an Agentic Document Understanding System.

Evaluate the following document analysis output based on these criteria:
1. Summary accuracy
2. Summary clarity
3. Keyword quality
4. Classification accuracy
5. Completeness

At the beginning of your response, write the final score exactly in this format:
Overall Score: X/5
Then give each criterion a score between 1 and 5.
Briefly explain each score.
Finally, provide 2 short improvement suggestions.

Document Text:
{limited_text}

Generated Summary:
{summary}

Extracted Keywords:
{", ".join(keywords)}

Predicted Category:
{category}
"""

        response = client.responses.create(
            model=model_name,
            input=prompt
        )

        return response.output_text

    except Exception as error:
        return f"LLM evaluation could not be completed. Error: {error}"


# -----------------------------
# 8. SIMILARITY ANALYSIS
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
# 9. AGENT DECISION LOGIC
# -----------------------------

def agent_decision(text, number_of_files, api_key):
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

    if api_key:
        decisions.append("API key detected. The system will run optional LLM-based evaluation.")
    else:
        decisions.append("No API key provided. The system will use rule-based evaluation only.")

    return decisions


# -----------------------------
# 10. REPORT GENERATION
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
    decisions,
    llm_evaluation,
    comparison_result
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

Rule-Based Evaluation:
- Score: {evaluation_score}/5
- Quality Label: {quality_label}

Rule-Based Evaluation Feedback:
"""

    for feedback in evaluation_feedback:
        report += f"- {feedback}\n"

    report += f"""

LLM-Based Evaluation:
{llm_evaluation}
    

    Rule-Based vs LLM Comparison Agent:
    - Rule-Based Score: {comparison_result["rule_based_score"]}/5
    - LLM Score: {comparison_result["llm_score"]}/5
    - Difference: {comparison_result["difference"]}
    - Same Result: {comparison_result["same_result"]}
    - Agreement Level: {comparison_result["agreement_level"]}
    - Agent Decision: {comparison_result["agent_decision"]}
    - Explanation: {comparison_result["explanation"]}
    - Recommendation: {comparison_result["recommendation"]}
    

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
# 11.EVALUATION COMPARISON AGENT
# -----------------------------

def extract_llm_score_on_5_scale(llm_evaluation_text):
    """
    Extract the overall LLM score from text and normalize it to a 5-point scale.
    Supported formats:
    - Overall Score: 4/5
    - Final Score: 8/10
    - Score: 80/100
    """

    if not llm_evaluation_text:
        return None

    text = str(llm_evaluation_text)

    error_keywords = [
        "skipped",
        "could not be completed",
        "error",
        "failed",
        "not found",
        "api key",
        "not installed"
    ]

    if any(keyword in text.lower() for keyword in error_keywords):
        return None

    patterns = [
        r"overall\s*score\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*/\s*(5|10|100)",
        r"final\s*score\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*/\s*(5|10|100)",
        r"total\s*score\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*/\s*(5|10|100)",
        r"score\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*/\s*(5|10|100)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            score = float(match.group(1))
            max_score = float(match.group(2))

            normalized_score = (score / max_score) * 5

            return round(normalized_score, 2)

    return None


def rule_based_vs_llm_agent(rule_based_score, llm_evaluation_text, tolerance=0.5):
    """
    Compare rule-based evaluation score and LLM-based evaluation score.
    Both scores are evaluated on a 5-point scale.
    """

    llm_score = extract_llm_score_on_5_scale(llm_evaluation_text)

    result = {
        "agent_name": "Rule-Based vs LLM Comparison Agent",
        "rule_based_score": rule_based_score,
        "llm_score": llm_score,
        "difference": None,
        "same_result": None,
        "agreement_level": None,
        "agent_decision": None,
        "explanation": None,
        "recommendation": None
    }

    if llm_score is None:
        result["same_result"] = "Unknown"
        result["agreement_level"] = "LLM score unavailable"
        result["agent_decision"] = "Comparison could not be completed"
        result["explanation"] = (
            "The rule-based score was generated, but the LLM evaluation did not return a valid numeric score."
        )
        result["recommendation"] = (
            "Make sure the LLM output starts with a clear score such as 'Overall Score: 4/5'."
        )
        return result

    difference = abs(rule_based_score - llm_score)
    result["difference"] = round(difference, 2)

    if difference == 0:
        result["same_result"] = "Yes"
        result["agreement_level"] = "Exact match"
        result["agent_decision"] = "Both methods produced the same result"
        result["explanation"] = (
            f"Both rule-based evaluation and LLM evaluation gave the same score: {rule_based_score}/5."
        )
        result["recommendation"] = (
            "The result is highly reliable because both evaluation methods agree."
        )

    elif difference <= tolerance:
        result["same_result"] = "Almost same"
        result["agreement_level"] = "High agreement"
        result["agent_decision"] = "Both methods produced very similar results"
        result["explanation"] = (
            f"Rule-based score is {rule_based_score}/5 and LLM score is {llm_score}/5. "
            f"The difference is {result['difference']}, which is within the tolerance limit."
        )
        result["recommendation"] = (
            "The result can be accepted. Small differences are normal because LLM evaluation is more flexible."
        )

    elif difference <= 1.5:
        result["same_result"] = "Partially"
        result["agreement_level"] = "Medium agreement"
        result["agent_decision"] = "The methods are partially aligned"
        result["explanation"] = (
            f"Rule-based score is {rule_based_score}/5 and LLM score is {llm_score}/5. "
            f"The difference is {result['difference']}, so the results are close but not identical."
        )
        result["recommendation"] = (
            "Manual review is recommended, or the rule-based criteria can be improved."
        )

    else:
        result["same_result"] = "No"
        result["agreement_level"] = "Low agreement"
        result["agent_decision"] = "The methods produced different results"
        result["explanation"] = (
            f"Rule-based score is {rule_based_score}/5 and LLM score is {llm_score}/5. "
            f"The difference is {result['difference']}, which shows a significant disagreement."
        )
        result["recommendation"] = (
            "Manual review is strongly recommended. The rule-based evaluator may be too strict, "
            "or the LLM may be interpreting the document differently."
        )

    return result

# -----------------------------
# 12. STREAMLIT INTERFACE
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
            # -----------------------------
            # RULE-BASED SUMMARY
            # -----------------------------
            rule_based_summary = generate_simple_summary(clean_text)
            # Existing summary variable is kept for rule-based evaluation and report compatibility
            summary = rule_based_summary

            # -----------------------------
            # LLM-BASED SUMMARY
            # -----------------------------
            llm_summary, llm_summary_status = generate_llm_summary(
                openai_api_key,
                llm_model,
                clean_text
            )

           
            keywords = extract_keywords(clean_text)
            category, category_reason = classify_document(clean_text)

            evaluation_score, evaluation_feedback = evaluate_output(
                clean_text,
                summary,
                keywords,
                category
            )
            decisions = agent_decision(clean_text, len(uploaded_files), openai_api_key)
            quality_label = get_quality_label(evaluation_score)

            llm_evaluation = generate_llm_evaluation(
                openai_api_key,
                llm_model,
                clean_text,
                summary,
                keywords,
                category
            )

            processed_documents.append({
                "file_name": uploaded_file.name,
                "clean_text": clean_text,
                "rule_based_summary": rule_based_summary,
                "llm_summary": llm_summary if llm_summary else "LLM summary not available.",
                "summary": summary,
                "keywords": keywords,
                "category": category,
                "category_reason": category_reason,
                "evaluation_score": evaluation_score,
                "evaluation_feedback": evaluation_feedback,
                "decisions": decisions,
                "llm_evaluation": llm_evaluation
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
                st.write("### Rule-Based Summary")
                st.info(rule_based_summary)
            
                st.write("### LLM-Based Summary")
                
            with col2:
                st.write("### Extracted Keywords")
                st.write(", ".join(keywords))

                st.write("### Rule-Based Evaluation Score")
                st.metric(
                    label="Output Quality",
                    value=f"{evaluation_score}/5",
                    delta=quality_label
                )
                
                for item in evaluation_feedback:
                    st.write(f"- {item}")

            

            if llm_summary:
                st.success(llm_summary_status)
                st.info(llm_summary)
            else:
                st.warning(llm_summary_status)
                    
            st.write("### LLM-Based Evaluation")
            if openai_api_key:
                st.success("LLM-based evaluation was generated.")
                st.write(llm_evaluation)
            else:
                st.warning("LLM evaluation skipped. Enter an OpenAI API key in the sidebar to enable it.")
                st.write(llm_evaluation)

            comparison_result = rule_based_vs_llm_agent(
                rule_based_score=evaluation_score,
                llm_evaluation_text=llm_evaluation,
                tolerance=0.5
            )

            st.write("### 🤖 Rule-Based vs LLM Comparison Agent")

            comp_col1, comp_col2, comp_col3 = st.columns(3)
            with comp_col1:
                st.metric(
                    "Rule-Based Score",
                    f"{comparison_result['rule_based_score']}/5"
                    if comparison_result["rule_based_score"] is not None
                    else "N/A"
                )

            with comp_col2:
                st.metric(
                    "LLM Score",
                    f"{comparison_result['llm_score']}/5"
                    if comparison_result["llm_score"] is not None
                    else "N/A"
                )

            with comp_col3:
                st.metric(
                    "Difference",
                    comparison_result["difference"]
                    if comparison_result["difference"] is not None
                    else "N/A"
                )

            st.write(f"**Same Result:** {comparison_result['same_result']}")
            st.write(f"**Agreement Level:** {comparison_result['agreement_level']}")
            st.write(f"**Agent Decision:** {comparison_result['agent_decision']}")

            st.info(comparison_result["explanation"])
            st.write(f"**Recommendation:** {comparison_result['recommendation']}")

            if comparison_result["same_result"] == "Yes":
                st.success("Rule-based and LLM evaluations give the same result.")
            elif comparison_result["same_result"] == "Almost same":
                st.success("Rule-based and LLM evaluations are very similar.")
            elif comparison_result["same_result"] == "Partially":
                st.warning("Rule-based and LLM evaluations are partially similar.")
            elif comparison_result["same_result"] == "No":
                st.error("Rule-based and LLM evaluations are different.")
            else:
                st.warning("Comparison could not be completed.")

            report_text = generate_analysis_report(
                uploaded_file.name,
                clean_text,
                summary,
                keywords,
                category,
                category_reason,
                evaluation_score,
                evaluation_feedback,
                decisions,
                llm_evaluation,
                comparison_result
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