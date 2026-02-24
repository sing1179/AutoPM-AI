# AutoPM AI

**An AI-powered product prioritization assistant that helps teams figure out *what* to build, not just how.**

Download, add your API key, and run. No sample data included—use your own customer interviews and usage data.

---

## Product Description

AutoPM AI addresses a critical gap in product development: most tools focus on execution (how to build), but product managers spend the majority of their time on discovery and prioritization (what to build). Customer interviews, usage analytics, support tickets, and feedback live in scattered documents and spreadsheets—synthesizing them into actionable product direction is manual, inconsistent, and time-consuming.

AutoPM AI changes that. Upload your customer interview transcripts, product usage data, and any relevant documents. Ask "What should we build next?" and get AI-powered, evidence-backed recommendations that cite specific findings from your data. Each recommendation includes priority, impact, and the exact evidence that supports it.

### The Problem

- **Customer interviews are underutilized** — Raw transcripts and notes rarely get structured, searchable, or reusable across the team.
- **"What to build" is under-served** — PMs juggle qualitative and quantitative signals with no unified way to synthesize them.
- **Specs and PRDs are brittle** — Static documents don't stay in sync with reality or with how teams actually work.

### The Solution

AutoPM AI combines customer feedback and usage data into a single analysis. It extracts pain points, feature requests, and themes from interviews; identifies patterns, drop-offs, and opportunities from usage data; and produces prioritized recommendations with cited evidence. The way we define and communicate "what to build" can finally evolve.

---

## Features

- **Multi-format upload** — Accepts customer interviews (.txt, .md, .pdf, .docx), usage data (.csv), and any text-based file (e.g. .json, .xml).
- **AI-powered analysis** — Uses Groq's Llama 3.3 70B model for fast, high-quality recommendations.
- **Evidence-backed output** — Each recommendation cites specific data from your uploaded files.
- **Simple interface** — Streamlit web app: upload, ask, get results.

---

## Quick Start

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/keys) (free tier available)

### Installation

```bash
git clone https://github.com/sing1179/AutoPM-AI.git
cd AutoPM-AI

python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

### Configuration

**Important:** This app requires your own API key. No keys are included or shared.

1. Copy the example env file: `cp .env.example .env`
2. Edit `.env` and add your [Groq API key](https://console.groq.com/keys) (free tier available):

```
GROQ_API_KEY=your_groq_api_key_here
```

3. **Never commit `.env`** — It is gitignored. Your key stays local.

### Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Deploy for Others

Deploy the app so anyone can use it:

- **[Streamlit Community Cloud](https://share.streamlit.io)** — Connect GitHub, deploy in one click. Free.
- **[Hugging Face Spaces](https://huggingface.co/spaces)** — Free hosting for AI apps.

See [DEPLOY.md](DEPLOY.md) for step-by-step instructions.

---

## Project Structure

```
AutoPM-AI/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template (copy to .env, add your own key)
├── .gitignore
└── README.md
```

---

## Usage

1. **Upload files** — Drag and drop your customer interview transcripts, usage CSVs, PDFs, or Word documents.
2. **Preview** — Expand each file to verify its contents.
3. **Ask** — Type your question (e.g. "What should we build next?") or use the default.
4. **Get recommendations** — Click "Get Recommendations" to receive 3–5 prioritized items with evidence.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | Groq (Llama 3.3 70B Versatile) |
| Document parsing | pypdf, python-docx, pandas |
| API client | OpenAI-compatible (Groq) |

---

## Dependencies

- `streamlit` — Web interface
- `openai` — OpenAI-compatible API client (used with Groq)
- `python-dotenv` — Environment variable loading
- `pandas` — CSV handling
- `pypdf` — PDF text extraction
- `python-docx` — Word document parsing

---

## License

MIT License

---

## Author

**Sole author and maintainer** — This project was built independently by a single developer.
