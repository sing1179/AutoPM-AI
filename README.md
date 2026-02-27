# AutoPM-AI

Product prioritization assistant. Upload customer interviews and usage data, get AI-powered recommendations and implementation-ready specs.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → Repository: your-repo, Branch: main, **Main file path: `app.py`**
4. Add secret: `GROQ_API_KEY`

## Requirements

- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com))
