"""
AutoPM AI - Product prioritization assistant
Upload customer interviews and usage data, get AI-powered recommendations and implementation-ready specs.
"""

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

from spec_schema import extract_spec_from_response, spec_dict_to_markdown
from agents import orchestrate_chat, orchestrate_spec

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
    api_key = os.getenv("GROQ_API_KEY")
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


def wants_spec(prompt: str) -> bool:
    """Detect if user is asking for an implementation spec."""
    p = prompt.lower().strip()
    triggers = [
        "generate spec", "create spec", "write spec", "spec for",
        "implementation spec", "implementation plan", "dev spec",
        "break down", "break this down", "task list", "dev tasks",
        "for coding", "ready for implementation",
    ]
    return any(t in p for t in triggers)


# --- UI ---
# Minimal CSS: background, typography, colors only. Layout via Streamlit primitives.

st.markdown("""
<style>
    .stApp { background: #000000 !important; }
    .stApp::before {
        content: ''; position: fixed; inset: 0;
        background-image: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 40px 40px; pointer-events: none; z-index: 0;
    }
    .main .block-container { position: relative; z-index: 10; max-width: 896px; padding: 1rem 2rem 2rem !important; }
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.6) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label { color: rgba(255,255,255,0.9) !important; }
    [data-testid="stSidebar"] .stCaption { color: rgba(255,255,255,0.6) !important; }
    .main .stMarkdown, .main label, .main p { color: #fff !important; }
    .main .stMarkdown h1, .main .stMarkdown h2, .main .stMarkdown h3, .main .stMarkdown h4 { color: #fff !important; }
    .main .stMarkdown li, .main .stMarkdown span { color: #fff !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Nav: Streamlit columns instead of raw HTML
nav = st.container()
with nav:
    col_logo, col_spacer, col_actions = st.columns([1, 4, 2])
    with col_logo:
        st.markdown("**AutoPM-AI**")
    with col_actions:
        st.caption("Research notes · Login · Create account")

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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_spec" not in st.session_state:
    st.session_state.last_spec = None
if "last_critique" not in st.session_state:
    st.session_state.last_critique = None

if not st.session_state.messages:
    hero = st.container()
    with hero:
        st.markdown("# Optimized for Thought  \n# Built for Action")
        st.caption("Think smarter and act faster, from idea to execution in seconds.")

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("critique"):
            with st.expander("🔍 Agent review (critic feedback)"):
                st.markdown(msg["critique"])
        if msg["role"] == "assistant" and i == len(st.session_state.messages) - 1:
            spec_dict = extract_spec_from_response(msg["content"])
            if spec_dict:
                st.session_state.last_spec = spec_dict
                st.caption("📋 Implementation-ready spec detected")
                col1, col2, col3 = st.columns(3)
                with col1:
                    md = spec_dict_to_markdown(spec_dict)
                    st.download_button("📥 Download .md", md, file_name="spec.md", mime="text/markdown", key="dl_md")
                with col2:
                    st.download_button("📥 Download .json", json.dumps(spec_dict, indent=2), file_name="spec.json", mime="application/json", key="dl_json")
                with col3:
                    with st.expander("📄 View spec"):
                        st.code(md, language="markdown")

if not st.session_state.messages:
    trusted = st.container()
    with trusted:
        st.caption("Trusted by")
        t1, t2, t3, t4 = st.columns(4)
        with t1: st.markdown("**Atlassian**")
        with t2: st.markdown("**Notion**")
        with t3: st.markdown("**Linear**")
        with t4: st.markdown("**GitHub**")

input_section = st.container()
with input_section:
    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_area(
            "Message",
            placeholder="Ask for anything or use a command",
            label_visibility="collapsed",
            height=100,
            key="chat_input",
        )
        submitted = st.form_submit_button("Generate")

    col_upload, col_feedback, _ = st.columns([2, 1, 3])
    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=None,
            accept_multiple_files=True,
            help="Interviews (.txt, .md, .pdf, .docx) • Usage data (.csv)",
            key="file_upload",
        )
    with col_feedback:
        st.button("💬 Feedback", key="feedback_btn")

if submitted and prompt and prompt.strip():
    prompt = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last_spec = None

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        client = get_llm_client()
        if not client:
            st.error("API key required. Add your Groq API key in the sidebar (or set GROQ_API_KEY in .env).")
        else:
            data_context = build_context(uploaded_files) if uploaded_files else None
            conv_for_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

            try:
                spinner_msg = "Generating implementation spec (3 agents)..." if wants_spec(prompt) else "Thinking (3 agents: analyst → critic → reviser)..."
                with st.spinner(spinner_msg):
                    if wants_spec(prompt):
                        response, critique = orchestrate_spec(client, conv_for_api, data_context)
                    else:
                        response, critique = orchestrate_chat(client, conv_for_api, data_context)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response, "critique": critique})
            except ValueError as e:
                st.error(str(e))
                st.info("Add your Groq API key in the sidebar (expand → Settings) or set GROQ_API_KEY in .env")
            except Exception as e:
                st.error(f"An error occurred: {e}")
