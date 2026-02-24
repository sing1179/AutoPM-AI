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
# Redefined Glass Page Design - Glassmorphism style

st.markdown("""
<style>
    /* Base: dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #0c4a6e 50%, #1e3a5f 75%, #0f172a 100%);
        background-attachment: fixed;
    }
    
    /* Glass sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    [data-testid="stSidebar"] .stCaption {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    
    /* Hero section - glass card */
    .hero-glass {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .hero-glass h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f8fafc !important;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .hero-glass p {
        color: rgba(248, 250, 252, 0.8) !important;
        font-size: 1.1rem;
    }
    
    /* File uploader - glass container */
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2.5rem;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploader"]:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(148, 163, 184, 0.3);
    }
    
    div[data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
    }
    
    /* Expanders - glass cards */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Recommendations output - glass card */
    .glass-recommendations {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
    }
    
    .glass-recommendations h3 {
        color: #f8fafc !important;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    
    .glass-recommendations .rec-content {
        color: rgba(248, 250, 252, 0.95);
        line-height: 1.6;
    }
    
    .glass-recommendations .rec-content h1, .glass-recommendations .rec-content h2, .glass-recommendations .rec-content h3 {
        color: #f8fafc !important;
    }
    
    .glass-recommendations .rec-content ul, .glass-recommendations .rec-content ol {
        margin: 0.5rem 0;
    }
    
    /* Main content text - light for dark bg */
    .main .stMarkdown {
        color: rgba(248, 250, 252, 0.95) !important;
    }
    
    .main .stMarkdown h1, .main .stMarkdown h2, .main .stMarkdown h3 {
        color: #f8fafc !important;
    }
    
    .main label {
        color: rgba(248, 250, 252, 0.9) !important;
    }
    
    /* Primary button - glass style */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.4) 0%, rgba(14, 165, 233, 0.4) 100%) !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(56, 189, 248, 0.5);
        color: #f8fafc !important;
        font-weight: 600;
        border-radius: 12px;
        transition: all 0.2s ease;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.5) 0%, rgba(14, 165, 233, 0.5) 100%) !important;
        border-color: rgba(56, 189, 248, 0.7);
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.2);
    }
    
    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Success / info / warning - glass-friendly */
    [data-testid="stAlert"] {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }
    
    /* Text input - glass style */
    .stTextInput input {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #f8fafc !important;
        border-radius: 10px;
    }
    
    .stTextInput input::placeholder {
        color: rgba(248, 250, 252, 0.5);
    }
    
    /* Expander content */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
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

# Hero - Glass card
st.markdown(
    '<div class="hero-glass"><h3 style="margin:0;font-size:2rem;">📋 AutoPM AI</h3>'
    '<p style="margin:0.5rem 0 0 0;opacity:0.9;">Upload customer interviews and usage data. Ask what to build next.</p></div>',
    unsafe_allow_html=True,
)

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

                import markdown
                rec_html = markdown.markdown(recommendations, extensions=["fenced_code", "tables"])
                st.markdown(
                    f'<div class="glass-recommendations"><h3>📌 Recommendations</h3><div class="rec-content">{rec_html}</div></div>',
                    unsafe_allow_html=True,
                )

            except ValueError as e:
                st.error(str(e))
                st.info("Add your Groq API key in the sidebar (or set GROQ_API_KEY in .env for local use)")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.caption("Check your API key and network connection. If the issue persists, try again later.")
