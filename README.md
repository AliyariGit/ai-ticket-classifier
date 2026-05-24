# 🎫 AI Ticket Classifier

An intelligent IT support ticket classification system powered by **local LLMs (Ollama/Phi-3)** with NLM-based categorization, priority scoring, sentiment analysis, and a real-time analytics dashboard.

![Dashboard Preview](docs/demo.gif)

---

## 🚀 Features

- 🤖 **LLM Classification** — Phi-3 via Ollama for intelligent ticket analysis
- 🔄 **Rule-based fallback** — works even without Ollama running
- 📊 **Real-time dashboard** — live analytics with category/priority/sentiment breakdown
- 🚨 **Priority scoring** — Critical / High / Medium / Low with SLA guidance
- 💬 **Sentiment detection** — Frustrated / Neutral / Satisfied
- 🎯 **Action suggestions** — AI recommends next steps for each ticket
- ⚡ **Batch classification** — process multiple tickets at once via API

---

## 🏗 Architecture

```
Ticket Text
     │
     ▼
┌─────────────┐     ┌─────────────────────┐
│  FastAPI    │────▶│  Ticket Classifier  │
│  REST API   │     │  (LLM + Fallback)   │
└─────────────┘     └─────────────────────┘
     │                        │
     │                ┌───────┴──────────┐
     │                │  Ollama (Phi-3)  │
     │                │  or Rule Engine  │
     │                └──────────────────┘
     ▼
┌─────────────────┐
│  Dashboard UI   │
│  (Analytics)    │
└─────────────────┘
```

---

## 📋 Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) *(optional — falls back to rule-based)*

```bash
# Optional: pull LLM model for AI-powered classification
ollama pull phi3
```

---

## ⚙️ Setup

```bash
git clone https://github.com/rezaaliyari/ai-ticket-classifier.git
cd ai-ticket-classifier
pip install -r backend/requirements.txt
```

### Start the API

```bash
cd backend
uvicorn api:app --reload --port 8001
```

### Open the Dashboard

```bash
cd frontend
python -m http.server 3001
# Visit http://localhost:3001
```

---

## 🎮 Usage

### Via Dashboard
1. Enter a ticket description in the **Classify a Ticket** panel
2. Hit **Classify Ticket** — results appear instantly
3. View analytics in the **Analytics** tab

### Via API

```bash
# Classify a single ticket
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "VPN is down and my whole team cannot work!", "ticket_id": "TKT-010"}'

# Get analytics
curl http://localhost:8001/analytics

# Batch classify
curl -X POST http://localhost:8001/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"tickets": [{"text": "Printer not working"}, {"text": "Password reset needed"}]}'
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/classify` | Classify a single ticket |
| `POST` | `/classify/batch` | Classify multiple tickets |
| `GET` | `/tickets` | List classified tickets |
| `GET` | `/analytics` | Get aggregate analytics |
| `DELETE` | `/tickets/reset` | Clear all tickets |

---

## 📊 Classification Output

```json
{
  "ticket_id": "TKT-001",
  "category": "Network & Connectivity",
  "priority": "Critical",
  "sentiment": "Frustrated",
  "summary": "VPN authentication failing, blocking remote team access.",
  "suggested_action": "Escalate to network team immediately; check auth server status.",
  "confidence": 0.92,
  "keywords": ["vpn", "authentication", "failing"],
  "method": "LLM",
  "classified_at": "2026-05-23T10:30:00"
}
```

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Ollama (Phi-3) |
| Fallback | Keyword rule engine |
| API | FastAPI |
| Frontend | Vanilla JS / HTML / CSS |

---

## 👤 Author

**Reza (Ray) Aliyari**
[linkedin.com/in/rezaaliyari](https://linkedin.com/in/rezaaliyari)

---

## 📄 License

MIT License
