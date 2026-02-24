"""
AutoPM-AI API - Backend for React frontend
Provides recommendations via Groq API.
"""

import os
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AutoPM-AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USAGE_EXTENSIONS = {".csv"}
PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}


def get_file_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def read_pdf_content(content: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(content))
        text_parts = [p.extract_text() for p in reader.pages if p.extract_text()]
        return "\n\n".join(text_parts) if text_parts else "[No text could be extracted from PDF]"
    except Exception as e:
        return f"[Error reading PDF: {e}]"


def read_docx_content(content: bytes) -> str:
    try:
        from docx import Document
        doc = Document(BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        return f"[Error reading Word document: {e}]"


def read_csv_as_text(content: bytes) -> str:
    try:
        import pandas as pd
        df = pd.read_csv(BytesIO(content))
        return df.to_string(index=False)
    except Exception as e:
        return f"[Error parsing CSV: {e}]"


def read_file_content(content: bytes, filename: str) -> str:
    ext = get_file_extension(filename)
    if ext in USAGE_EXTENSIONS:
        return read_csv_as_text(content)
    if ext in PDF_EXTENSIONS:
        return read_pdf_content(content)
    if ext in DOCX_EXTENSIONS:
        return read_docx_content(content)
    try:
        return content.decode("utf-8", errors="replace")
    except Exception as e:
        return f"[Error reading file: {e}]"


def build_context(files: list[tuple[str, bytes]]) -> str:
    parts = []
    for filename, content in files:
        text = read_file_content(content, filename)
        parts.append(f"--- FILE: {filename} ---\n{text}")
    return "\n\n".join(parts)


def get_recommendations(context: str, user_question: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.strip() == "" or "your_groq_api_key" in api_key.lower():
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    client = OpenAI(api_key=api_key.strip(), base_url="https://api.groq.com/openai/v1")

    system_prompt = """You are an expert product manager analyzing customer feedback and usage data to recommend what to build next.

Your task is to:
1. Analyze customer interviews and documents for: pain points, feature requests, recurring themes, and unmet needs
2. Analyze usage data (from .csv files) for: usage patterns, drop-offs, underutilized features, and growth opportunities
3. Synthesize both data sources to produce 3-5 prioritized "what to build next" recommendations

For each recommendation:
- Provide a clear, actionable title
- Cite specific evidence from the uploaded data
- Explain the impact/opportunity
- Suggest a priority (High/Medium/Low) with justification

Format your response in clean markdown with headers and bullet points. Be concise but thorough."""

    user_prompt = f"## Uploaded Data\n\n{context}\n\n## User Question\n\n{user_question}\n\nPlease analyze the above data and provide your prioritized recommendations."

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


@app.post("/api/recommendations")
async def recommendations(
    query: str = Form(...),
    files: list[UploadFile] = File(default=[]),
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    file_contents = []
    for f in files:
        content = await f.read()
        file_contents.append((f.filename or "unknown", content))

    context = build_context(file_contents)

    try:
        result = get_recommendations(context, query)
        return {"recommendations": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "ok"}
