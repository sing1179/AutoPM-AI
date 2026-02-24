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

# Page config - match Figma: full width, no sidebar by default
st.set_page_config(
    page_title="AutoPM-AI - Optimized for Thought, Built for Action",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
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
# Figma design: black bg, dotted grid, amber/orange orbs, nav bar, hero, glass card

st.markdown("""
<style>
    /* Base: black background with dotted grid */
    .stApp {
        background: #000000 !important;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image: radial-gradient(circle, rgba(255,255,255,0.08) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Glowing orbs - use injected divs (see HTML below) */
    
    /* Ensure content is above orbs */
    .main .block-container {
        position: relative;
        z-index: 10;
    }
    
    /* Glass sidebar */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.6) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label { color: rgba(255,255,255,0.9) !important; }
    [data-testid="stSidebar"] .stCaption { color: rgba(255,255,255,0.6) !important; }
    
    /* Main glass card - middle column in first columns layout */
    .figma-glass-card {
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        padding: 2rem;
        margin: 1rem 0;
    }
    
    #main-glass-card ~ div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) {
        background: rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        padding: 2rem !important;
    }
    
    
    /* File uploader - glass style */
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    div[data-testid="stFileUploader"] section { background: transparent !important; border: none !important; }
    
    /* Text input - dark glass */
    .stTextInput input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #fff !important;
        border-radius: 12px;
    }
    
    .stTextInput input::placeholder { color: rgba(255,255,255,0.4); }
    
    /* Primary button - Generate style */
    .stButton > button[kind="primary"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #fff !important;
        font-weight: 500;
        border-radius: 0.5rem;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(255, 255, 255, 0.25);
    }
    
    /* Secondary buttons */
    .stButton > button:not([kind="primary"]) {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 0.5rem;
    }
    
    .stButton > button:hover:not([kind="primary"]) {
        background: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Main content text */
    .main .stMarkdown, .main label { color: rgba(255,255,255,0.95) !important; }
    .main .stMarkdown h1, .main .stMarkdown h2, .main .stMarkdown h3 { color: #fff !important; }
    
    /* Recommendations output */
    .glass-recommendations {
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(24px);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .glass-recommendations h3 { color: #fff !important; margin: 0 0 1rem 0; }
    .glass-recommendations .rec-content { color: rgba(255,255,255,0.9); line-height: 1.6; }
    .glass-recommendations .rec-content h1, .glass-recommendations .rec-content h2, .glass-recommendations .rec-content h3 { color: #fff !important; }
    
    /* Alerts */
    [data-testid="stAlert"] {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }
    
    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Glowing orbs (behind content)
st.markdown("""
<style>@keyframes orb-pulse { 0%,100%{opacity:0.8} 50%{opacity:1} }</style>
<div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); 
    width: 800px; height: 800px; background: rgba(245, 158, 11, 0.25); border-radius: 50%; 
    filter: blur(120px); pointer-events: none; z-index: 0; animation: orb-pulse 4s ease-in-out infinite;"></div>
<div style="position: fixed; bottom: 0; right: 0; width: 600px; height: 600px; 
    background: rgba(251, 146, 60, 0.2); border-radius: 50%; filter: blur(100px); 
    pointer-events: none; z-index: 0;"></div>
""", unsafe_allow_html=True)

# Nav bar
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; padding: 1.5rem 2rem; margin: -1rem -1rem 1rem -1rem;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="width: 36px; height: 36px; background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; color: white;">AI</div>
        <span style="color: white; font-size: 1.125rem; font-weight: 600;">AutoPM-AI</span>
    </div>
    <div style="display: flex; align-items: center; gap: 2rem;">
        <span style="color: rgba(255,255,255,0.7); font-size: 0.875rem; cursor: pointer;">Research notes</span>
        <button style="padding: 0.5rem 1rem; border-radius: 0.5rem; border: 1px solid rgba(255,255,255,0.2); background: transparent; color: white; font-size: 0.875rem; cursor: pointer;">Login</button>
        <button style="padding: 0.5rem 1rem; border-radius: 0.5rem; background: white; color: black; font-size: 0.875rem; font-weight: 500; cursor: pointer; border: none;">Create account</button>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar - API key (collapsed by default, user can expand)
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.caption("API key required for AI recommendations.")
    if not get_api_key():
        api_key_input = st.text_input(
            "Groq API key",
            type="password",
            placeholder="Paste your key here",
            key="groq_key_input",
        )
        if api_key_input:
            st.session_state.groq_api_key = api_key_input
            st.rerun()
    else:
        st.success("✓ Ready")
    st.caption("Formats: .txt, .md, .pdf, .docx, .csv")

# Hero
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem 0;">
    <h1 style="font-size: 3.75rem; font-weight: 700; color: white; font-family: system-ui, sans-serif; line-height: 1.1; letter-spacing: -0.025em; margin: 0;">
        Optimized for Thought<br/>Built for Action
    </h1>
    <p style="color: rgba(255,255,255,0.6); font-size: 1.125rem; margin-top: 1.5rem;">
        Think smarter and act faster, from idea to execution in seconds.
    </p>
</div>
""", unsafe_allow_html=True)

# Main glass card - centered (ID for CSS targeting)
st.markdown('<div id="main-glass-card"></div>', unsafe_allow_html=True)
col_left, col_center, col_right = st.columns([1, 4, 1])
with col_center:
    # AI input + Generate button
    col1, col2 = st.columns([5, 1])
    with col1:
        user_question = st.text_input(
            "Ask your question",
            value="What should we build next?",
            placeholder="Ask for anything or use a command",
            label_visibility="collapsed",
            key="query_input",
        )
    with col2:
        st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("Generate", type="primary", use_container_width=True)

    # Upload and Feedback - same row
    col_upload, col_feedback = st.columns([1, 1])
    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=None,
            accept_multiple_files=True,
            help="Interviews (.txt, .md, .pdf, .docx) • Usage data (.csv)",
            key="file_upload",
        )
    with col_feedback:
        st.markdown("<div style='height: 52px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <a href="#" style="display: inline-flex; align-items: center; gap: 8px; padding: 0.5rem 1rem; 
           border-radius: 0.5rem; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); 
           color: rgba(255,255,255,0.7); font-size: 0.875rem; text-decoration: none;">
            💬 Feedback
        </a>
        """, unsafe_allow_html=True)

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
elif not uploaded_files and not analyze_clicked:
    st.markdown(
        '<p style="text-align: center; color: rgba(255,255,255,0.4); font-size: 0.875rem; padding: 2rem;">'
        'Upload documents to get AI-powered recommendations</p>',
        unsafe_allow_html=True,
    )

# Results
if analyze_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one file before analyzing.")
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
                st.info("Add your Groq API key in the sidebar (expand → Settings) or set GROQ_API_KEY in .env")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Trusted by section
st.markdown("""
<div style="text-align: center; padding: 3rem 0 2rem 0;">
    <p style="color: rgba(255,255,255,0.4); font-size: 0.875rem; margin-bottom: 1.5rem;">Trusted by</p>
    <div style="display: flex; align-items: center; justify-content: center; gap: 3rem; opacity: 0.4;">
        <span style="color: white; font-size: 1.125rem; font-weight: 600;">Atlassian</span>
        <span style="color: white; font-size: 1.125rem; font-weight: 600;">Notion</span>
        <span style="color: white; font-size: 1.125rem; font-weight: 600;">Linear</span>
        <span style="color: white; font-size: 1.125rem; font-weight: 600;">GitHub</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Help button - bottom right
st.markdown("""
<div style="position: fixed; bottom: 24px; right: 24px; width: 40px; height: 40px; 
    background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.1); 
    border-radius: 50%; display: flex; align-items: center; justify-content: center; 
    font-size: 1.25rem; color: rgba(255,255,255,0.7); cursor: pointer; z-index: 9999;">?</div>
""", unsafe_allow_html=True)
