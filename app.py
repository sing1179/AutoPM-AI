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


CONVERSATION_SYSTEM = """You are a sharp, no-nonsense PM assistant. Your job is to give straight answers and move the conversation forward.

RULES:
1. **Be direct** — Answer what the PM actually asked. No preamble, no fluff.
2. **Ask clarifying questions when needed** — If the question is vague, ambiguous, or you need context (e.g. "which users?", "what's the timeline?"), ask 1–2 short clarifying questions. Don't guess.
3. **Use the data when you have it** — If documents are uploaded, cite specific evidence. If not, say so and ask them to upload.
4. **Match the ask** — If they want prioritization, give priorities. If they want a summary, summarize. If they want tradeoffs, give tradeoffs.
5. **Keep it concise** — Bullet points over paragraphs. Markdown when helpful.
6. **Never lecture** — Skip the "As a product manager..." stuff. Just help."""


def chat_turn(messages: list[dict], data_context: str | None) -> str:
    """Call Groq with full conversation history. Returns assistant response."""
    client = get_llm_client()
    if not client:
        raise ValueError("API key required. Add your Groq API key in the sidebar (or set GROQ_API_KEY in .env).")

    system = CONVERSATION_SYSTEM
    if data_context:
        system += "\n\n## Available data\nYou have access to the following uploaded documents. Use them when relevant.\n\n" + data_context
    else:
        system += "\n\n## Available data\nNone. If the user asks for analysis or recommendations, ask them to upload documents first."

    api_messages = [{"role": "system", "content": system}]
    for m in messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=api_messages,
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

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hero - only when no messages yet
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1 style="font-size: 3.75rem; font-weight: 700; color: white; font-family: system-ui, sans-serif; line-height: 1.1; letter-spacing: -0.025em; margin: 0;">
            Optimized for Thought<br/>Built for Action
        </h1>
        <p style="color: rgba(255,255,255,0.6); font-size: 1.125rem; margin-top: 1.5rem;">
            Think smarter and act faster. Ask anything—or upload docs for analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

# File upload
uploaded_files = st.file_uploader(
    "Upload documents (optional—you can chat first)",
    type=None,
    accept_multiple_files=True,
    help="Interviews (.txt, .md, .pdf, .docx) • Usage data (.csv)",
    key="file_upload",
)
if uploaded_files:
    st.caption(f"✓ {len(uploaded_files)} file(s) attached to this conversation")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask anything—or upload docs for analysis"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        data_context = build_context(uploaded_files) if uploaded_files else None
        conv_for_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        try:
            with st.spinner("Thinking..."):
                response = chat_turn(conv_for_api, data_context)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
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
