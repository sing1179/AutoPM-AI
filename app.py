"""
AutoPM AI - Product prioritization assistant
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
    page_title="AutoPM AI - What to Build Next",
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


def get_api_key() -> str | None:
    """Get Groq API key from env, session state, or None if missing."""
    # 1. Environment variable (local dev, Streamlit Cloud secrets)
    api_key = os.getenv("GROQ_API_KEY")
    # 2. Session state (deployed app - user pastes key in sidebar)
    if not api_key and "groq_api_key" in st.session_state and st.session_state.groq_api_key:
        api_key = st.session_state.groq_api_key
    if not api_key or api_key.strip() == "" or "your_groq_api_key" in api_key.lower():
        return None
    return api_key.strip()


def get_llm_client() -> OpenAI | None:
    """Get Groq client (OpenAI-compatible), returns None if API key is missing."""
    api_key = get_api_key()
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


def get_recommendations(context: str, user_question: str) -> str:
    """Call Groq API to get product recommendations."""
    client = get_llm_client()
    if not client:
        raise ValueError("API key required. Add your Groq API key in the sidebar (or set GROQ_API_KEY in .env).")

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

# Custom CSS for polish
st.markdown("""
<style>
    .hero {
        padding: 1.5rem 0;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    .hero p {
        color: #64748b;
        font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        border: 1px solid #bae6fd;
        margin-bottom: 0.5rem;
    }
    .metric-card strong {
        color: #0369a1;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1;
        border-radius: 0.5rem;
        padding: 2rem;
        background: #f8fafc;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #2563eb;
        background: #fff;
    }
    .stMarkdown h3 { margin-top: 1.25rem; color: #1e293b; }
    .stMarkdown ul { margin: 0.5rem 0; }
    .stMarkdown li { margin: 0.25rem 0; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### AutoPM AI")
    st.caption("Figure out *what* to build next.")
    st.divider()
    if not get_api_key():
        st.caption("🔑 API Key")
        api_key_input = st.text_input(
            "Groq API key",
            type="password",
            placeholder="Paste your key here",
            help="Get a free key at console.groq.com",
            key="groq_key_input",
        )
        if api_key_input:
            st.session_state.groq_api_key = api_key_input
            st.rerun()
    else:
        st.success("✓ Ready")
    st.divider()
    st.caption("**Supported formats**")
    st.caption("Interviews: .txt, .md, .pdf, .docx")
    st.caption("Data: .csv")

# Hero
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.title("📋 AutoPM AI")
st.markdown("*Upload customer interviews and usage data. Ask what to build next.*")
st.markdown('</div>', unsafe_allow_html=True)

# File upload
uploaded_files = st.file_uploader(
    "**Drag and drop files here** — or browse",
    type=None,
    accept_multiple_files=True,
    help="Interviews (.txt, .md, .pdf, .docx) • Usage data (.csv)",
)

# File preview
if uploaded_files:
    st.success(f"✓ {len(uploaded_files)} file(s) ready")
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
else:
    st.info("👆 Upload customer interviews and/or usage data to get started")

# Question input
st.markdown("---")
st.subheader("💡 What should we build next?")

col1, col2 = st.columns([4, 1])

with col1:
    user_question = st.text_input(
        "Ask your question",
        value="What should we build next?",
        placeholder="What should we build next?",
        label_visibility="collapsed",
    )

with col2:
    analyze_clicked = st.button("Get Recommendations", type="primary", use_container_width=True)

st.markdown("---")

# Results
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
                st.info("Add your Groq API key in the sidebar (or set GROQ_API_KEY in .env for local use)")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.caption("Check your API key and network connection. If the issue persists, try again later.")
