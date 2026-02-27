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
# Redefined Glass Page Design (from Figma spec)

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
    [data-testid="stForm"] {
        background: rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        padding: 2rem !important;
        margin: 1rem 0 !important;
    }
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #fff !important;
        caret-color: #fff !important;
        border-radius: 16px !important;
    }
    .stTextArea textarea::placeholder { color: rgba(255,255,255,0.4) !important; }
    .stTextArea textarea:focus {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    .stButton > button {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #fff !important;
        border-radius: 12px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
    }
    div[data-testid="stFileUploader"] section { background: transparent !important; border: none !important; }
    .main .stMarkdown, .main label, .main p { color: #fff !important; }
    .main .stMarkdown h1, .main .stMarkdown h2, .main .stMarkdown h3, .main .stMarkdown h4 { color: #fff !important; }
    .main .stMarkdown li, .main .stMarkdown span { color: #fff !important; }
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1rem;
    }
    [data-testid="stChatMessage"] .stMarkdown, [data-testid="stChatMessage"] p { color: #fff !important; }
    [data-testid="stAlert"] {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
    [data-testid="stExpander"] .stMarkdown, [data-testid="stExpander"] p { color: #fff !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>@keyframes orb-pulse { 0%,100%{opacity:0.8} 50%{opacity:1} }</style>
<div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); 
    width: 800px; height: 800px; background: rgba(245, 158, 11, 0.4); border-radius: 50%; 
    filter: blur(120px); pointer-events: none; z-index: 1; animation: orb-pulse 4s ease-in-out infinite;"></div>
<div style="position: fixed; bottom: 0; right: 0; width: 600px; height: 600px; 
    background: rgba(251, 146, 60, 0.3); border-radius: 50%; filter: blur(100px); 
    pointer-events: none; z-index: 1;"></div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; padding: 1.5rem 2rem; margin: -1rem -1rem 1rem -1rem;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="width: 36px; height: 36px;">
            <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" style="width: 36px; height: 36px;">
                <circle cx="12" cy="16" r="10" fill="black" stroke="white" stroke-width="2"/>
                <text x="12" y="21" font-size="10" font-weight="bold" fill="white" text-anchor="middle">PM</text>
                <circle cx="20" cy="16" r="10" fill="black" stroke="white" stroke-width="2"/>
                <text x="20" y="21" font-size="10" font-weight="bold" fill="white" text-anchor="middle">AI</text>
            </svg>
        </div>
        <span style="color: white; font-size: 1.125rem; font-weight: 600;">AutoPM-AI</span>
    </div>
    <div style="display: flex; align-items: center; gap: 2rem;">
        <span style="color: rgba(255,255,255,0.7); font-size: 0.875rem; cursor: pointer;">Research notes</span>
        <button style="padding: 0.5rem 1rem; border-radius: 0.5rem; border: 1px solid rgba(255,255,255,0.2); background: transparent; color: white; font-size: 0.875rem; cursor: pointer;">Login</button>
        <button style="padding: 0.5rem 1rem; border-radius: 0.5rem; background: white; color: black; font-size: 0.875rem; font-weight: 500; cursor: pointer; border: none;">Create account</button>
    </div>
</div>
""", unsafe_allow_html=True)

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

with st.form("chat_form", clear_on_submit=True):
    col_logo, col_input = st.columns([0.5, 12])
    with col_logo:
        st.markdown("""
        <div style="padding-top: 1rem;">
            <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" style="width: 20px; height: 20px;">
                <circle cx="12" cy="16" r="10" fill="black" stroke="white" stroke-width="2"/>
                <text x="12" y="21" font-size="10" font-weight="bold" fill="white" text-anchor="middle">PM</text>
                <circle cx="20" cy="16" r="10" fill="black" stroke="white" stroke-width="2"/>
                <text x="20" y="21" font-size="10" font-weight="bold" fill="white" text-anchor="middle">AI</text>
            </svg>
        </div>
        """, unsafe_allow_html=True)
    with col_input:
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
    st.markdown("""
    <div style="padding-top: 0.5rem;">
        <button style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; 
            border-radius: 0.5rem; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); 
            color: rgba(255,255,255,0.7); font-size: 0.875rem; cursor: pointer;">
            <span style="font-size: 1rem;">💬</span> Feedback
        </button>
    </div>
    """, unsafe_allow_html=True)

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
