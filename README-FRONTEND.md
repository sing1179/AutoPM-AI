# AutoPM-AI React Frontend

Exact Figma design from Redefined Glass Page Design. Matches the screenshot with:
- Black background, dotted grid, amber/orange glowing orbs
- Nav bar (Logo, AutoPM-AI, Research notes, Login, Create account)
- Hero: "Optimized for Thought / Built for Action"
- Glass card with AI input, Generate, Upload documents, Feedback
- Trusted by section, Help (?) button

## Run the full stack

### 1. Start the API (backend)

```bash
cd AutoPM-AI
pip install -r requirements-api.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
uvicorn api.main:app --reload --port 8000
```

### 2. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Production build

```bash
cd frontend
npm run build
# Serve dist/ with your preferred static host (Vercel, Netlify, etc.)
# Point API to your deployed backend via VITE_API_URL
```
