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

# Initialize session state for quick actions
if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = "What should we build next?"

# Custom CSS - inspired by glassmorphism + Omnix aesthetic
# Use multiple selectors for compatibility across Streamlit versions
st.markdown("""
<style>
    /* Gradient background - multiple selectors for Streamlit compatibility */
    html, body, #root,
    section[data-testid="stApp"],
    div[data-testid="stApp"],
    [data-testid="stAppViewContainer"],
    .stApp {
        background: linear-gradient(135deg, #fef7ed 0%, #fce7f3 40%, #e0e7ff 100%) !important;
    }
    
    /* Block container - main content area */
    div.block-container {
        padding-top: 2rem !important;
        max-width: 900px !important;
    }
    
    /* Hero - centered, bold */
    .hero-box {
        text-align: center;
        padding: 2rem 0 2.5rem;
    }
    .hero-box h1 { font-size: 2.5rem !important; font-weight: 700 !important; margin-bottom: 0.5rem !important; }
    .hero-box .tagline { font-size: 1.15rem !important; color: #64748b !important; }
    
    /* Input card - glassmorphism */
    .input-card {
        background: rgba(255,255,255,0.85) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(139,92,246,0.2) !important;
        border-radius: 1rem !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 8px 32px rgba(139,92,246,0.15) !important;
    }
    
    /* Text input - pill shape */
    input[data-testid="stTextInput"] {
        border-radius: 2rem !important;
    }
    div[data-testid="stTextInput"] > div {
        border-radius: 2rem !important;
        background: rgba(248,250,252,0.95) !important;
    }
    
    /* Glowing purple Generate button */
    .stButton > button {
        border-radius: 0.75rem !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"],
    div[data-testid="column"] .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
        box-shadow: 0 0 20px rgba(139,92,246,0.4) !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 28px rgba(139,92,246,0.6) !important;
    }
    
    /* File uploader - dashed purple border */
    section[data-testid="stFileUploader"],
    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(139,92,246,0.5) !important;
        border-radius: 1rem !important;
        padding: 2rem !important;
        background: rgba(255,255,255,0.6) !important;
    }
    
    .stMarkdown h3 { margin-top: 1.25rem !important; }
    
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### AutoPM AI")
    st.caption("Optimized for thought. Built for action.")
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

# Hero - inline styles for reliability (CSS classes may not apply in all Streamlit versions)
st.markdown("""
<div style="text-align: center; padding: 2rem 0 2.5rem;">
    <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">📋 AutoPM AI</h1>
    <p style="font-size: 1.15rem; color: #64748b;">Think smarter. Act faster. From idea to execution.</p>
</div>
""", unsafe_allow_html=True)

# File upload
uploaded_files = st.file_uploader(
    "**+ Add files** — Drag and drop or browse",
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
    st.info("👆 Upload customer interviews and usage data to get started")

# Main input card
st.markdown("**✨ Ask for anything or use a command**")

# Quick action buttons (like Omnix suggested prompts)
quick_prompts = [
    "What should we build next?",
    "Prioritize features from feedback",
    "Identify top pain points",
]
cols = st.columns(len(quick_prompts))
for i, prompt in enumerate(quick_prompts):
    with cols[i]:
        if st.button(f"📌 {prompt}", key=f"quick_{i}", use_container_width=True):
            st.session_state.selected_prompt = prompt
            st.rerun()

# Pill-shaped input + Generate button
col1, col2 = st.columns([4, 1])
with col1:
    user_question = st.text_input(
        "Your question here",
        value=st.session_state.selected_prompt,
        placeholder="Ask for anything or use a command",
        label_visibility="collapsed",
        key="main_question",
    )
with col2:
    analyze_clicked = st.button("Generate →", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

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
