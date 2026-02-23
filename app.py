"""
PM Agent - Product prioritization assistant
Upload customer interviews and usage data, get AI-powered recommendations.
"""

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="PM Agent - What to Build Next",
    page_icon="📋",
    layout="wide",
)

# File type handling
USAGE_EXTENSIONS = {".csv"}
PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}


def get_file_extension(filename: str) -> str:
    """Get lowercase file extension."""
    return os.path.splitext(filename)[1].lower()


def read_pdf_content(uploaded_file) -> str:
    """Extract text from PDF file."""
    try:
        from pypdf import PdfReader
        uploaded_file.seek(0)
        reader = PdfReader(uploaded_file)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts) if text_parts else "[No text could be extracted from PDF]"
    except Exception as e:
        return f"[Error reading PDF: {e}]"


def read_docx_content(uploaded_file) -> str:
    """Extract text from Word (.docx) file."""
    try:
        from docx import Document
        from io import BytesIO
        uploaded_file.seek(0)
        doc = Document(BytesIO(uploaded_file.getvalue()))
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        return f"[Error reading Word document: {e}]"


def read_file_content(uploaded_file) -> str:
    """Read uploaded file content as string."""
    try:
        uploaded_file.seek(0)
        content = uploaded_file.getvalue().decode("utf-8", errors="replace")
        return content
    except Exception as e:
        return f"[Error reading file: {e}]"


def read_csv_as_text(uploaded_file) -> str:
    """Read CSV file and return as formatted text for context."""
    try:
        import pandas as pd
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        return df.to_string(index=False)
    except Exception as e:
        return f"[Error parsing CSV: {e}]"


def read_uploaded_file(uploaded_file) -> str:
    """Read any uploaded file and return text content."""
    ext = get_file_extension(uploaded_file.name)
    if ext in USAGE_EXTENSIONS:
        return read_csv_as_text(uploaded_file)
    if ext in PDF_EXTENSIONS:
        return read_pdf_content(uploaded_file)
    if ext in DOCX_EXTENSIONS:
        return read_docx_content(uploaded_file)
    # Fallback: try to decode as text (works for .txt, .md, .json, .xml, etc.)
    return read_file_content(uploaded_file)


def build_context(uploaded_files: list) -> str:
    """Combine all uploaded file contents into context string."""
    context_parts = []
    for uploaded_file in uploaded_files:
        content = read_uploaded_file(uploaded_file)
        context_parts.append(f"--- FILE: {uploaded_file.name} ---\n{content}")
    return "\n\n".join(context_parts)


def get_llm_client() -> OpenAI | None:
    """Get Groq client (OpenAI-compatible), returns None if API key is missing."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.strip() == "" or "your_groq_api_key" in api_key.lower():
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


def get_recommendations(context: str, user_question: str) -> str:
    """Call Groq API to get product recommendations."""
    client = get_llm_client()
    if not client:
        raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")

    system_prompt = """You are an expert product manager analyzing customer feedback and usage data to recommend what to build next.

Your task is to:
1. Analyze customer interviews and documents (from .txt, .md, .pdf, and other text-based files) for: pain points, feature requests, recurring themes, and unmet needs
2. Analyze usage data (from .csv files) for: usage patterns, drop-offs, underutilized features, and growth opportunities
3. Synthesize both data sources to produce 3-5 prioritized "what to build next" recommendations

For each recommendation:
- Provide a clear, actionable title
- Cite specific evidence from the uploaded data (quote or reference specific findings)
- Explain the impact/opportunity
- Suggest a priority (High/Medium/Low) with justification

Format your response in clean markdown with headers and bullet points. Be concise but thorough."""

    user_prompt = f"""## Uploaded Data

{context}

## User Question

{user_question}

Please analyze the above data and provide your prioritized recommendations."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


# --- UI ---

st.title("📋 PM Agent")
st.markdown("*Upload customer interviews and usage data. Ask what to build next.*")
st.divider()

# File upload section
st.subheader("📁 Upload Files")
st.caption("Accepts any file: .txt, .md, .pdf, .docx, .csv, .json, and more. Documents are extracted for analysis.")

uploaded_files = st.file_uploader(
    "Drag and drop or browse",
    type=None,  # Accept all file types
    accept_multiple_files=True,
)

# Display uploaded files
if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} file(s)")

    for uploaded_file in uploaded_files:
        ext = get_file_extension(uploaded_file.name)
        with st.expander(f"📄 {uploaded_file.name}", expanded=False):
            if ext in USAGE_EXTENSIONS:
                try:
                    import pandas as pd
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Could not parse CSV: {e}")
                    st.code(read_file_content(uploaded_file))
            else:
                content = read_uploaded_file(uploaded_file)
                st.text_area("Content", value=content, height=150, disabled=True, key=uploaded_file.name)

st.divider()

# Question input and analyze
st.subheader("💡 What should we build next?")

col1, col2 = st.columns([3, 1])

with col1:
    user_question = st.text_input(
        "Ask your question",
        value="What should we build next?",
        placeholder="What should we build next?",
        label_visibility="collapsed",
    )

with col2:
    analyze_clicked = st.button("Get Recommendations", type="primary", use_container_width=True)

st.divider()

# Results section
if analyze_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one file (customer interviews or usage data) before analyzing.")
    elif not user_question or not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing your data..."):
            try:
                context = build_context(uploaded_files)
                recommendations = get_recommendations(context, user_question.strip())

                st.subheader("📌 Recommendations")
                st.markdown(recommendations)

            except ValueError as e:
                st.error(str(e))
                st.info("Create a `.env` file in the PM_agent directory with: `GROQ_API_KEY=your_key_here`")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.caption("Check your API key and network connection. If the issue persists, try again later.")
