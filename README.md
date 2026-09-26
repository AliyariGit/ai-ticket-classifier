# 🎫 AI Ticket Classifier

An IT support ticket classification system powered by local LLMs, with priority scoring, sentiment analysis, rule-based fallback, and a real-time analytics dashboard. The optional provider is a QLoRA adapter on 4-bit Llama 3.2 3B, with local knowledge retrieval.

![Dashboard Preview](docs/demo.gif)

---

## 🚀 Features

- 🤖 **LLM Classification** — Phi-3 via Ollama by default, or optional QLoRA Llama 3.2 3B
- 📚 **Local retrieval** — optionally provide relevant knowledge from a JSONL file
- 🔄 **Rule-based fallback** — works even without Ollama running
- 📊 **Real-time dashboard** — live analytics with category/priority/sentiment breakdown
- 🚨 **Priority scoring** — Critical / High / Medium / Low with SLA guidance
- 💬 **Sentiment detection** — Frustrated / Neutral / Satisfied
- 🎯 **Action suggestions** — AI recommends next steps for each ticket
- ⚡ **Batch classification** — process multiple tickets at once via API

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────┐
│           BROWSER (port 3001)                │
│       index.html — Vanilla JS SPA            │
│  User submits ticket → fetch() → REST API    │
└──────────────────────┬───────────────────────┘
                       │  HTTP (JSON)
                       ▼
┌──────────────────────────────────────────────┐
│         FASTAPI SERVER (port 8001)           │
│  api.py — Routes, validation, in-memory DB   │
└──────────────────────┬───────────────────────┘
                       │  Python function call
                       ▼
┌──────────────────────────────────────────────┐
│          CLASSIFICATION ENGINE               │
│  classifier.py — provider selection,        │
│  context retrieval, validation, fallback     │
│   ┌───────────┐ ┌──────────────┐ ┌─────────┐ │
│   │ Ollama    │ │ 4-bit Llama  │ │ Rules   │ │
│   │ Phi-3     │ │ + LoRA       │ │ fallback│ │
│   └───────────┘ └──────────────┘ └─────────┘ │
│  retrieval.py — optional local JSONL search  │
└──────────────────────────────────────────────┘
```

For a full end-to-end walkthrough — including how FastAPI processes requests, how the LLM prompt is built, how the frontend fetches data, and how errors are handled — see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## 📋 Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) *(optional; current default provider)*
- CUDA-capable NVIDIA GPU and compatible PyTorch/CUDA/BitsAndBytes stack *(only for Llama training/inference)*

```bash
# Optional default provider: pull Ollama model
ollama pull phi3
```

---

## ⚙️ Setup

```bash
git clone https://github.com/AliyariGit/ai-ticket-classifier.git
cd ai-ticket-classifier
pip install -r backend/requirements.txt
```

The base install intentionally does not install GPU libraries. To use or train the Llama adapter, install a PyTorch build for your CUDA runtime using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the optional stack:

```bash
pip install -r backend/requirements-llama.txt
```

Llama 3.2 access may require accepting Meta's model license and authenticating with Hugging Face. CUDA/Linux is the recommended setup; BitsAndBytes support varies by platform.

### Start the API

```bash
cd backend
uvicorn api:app --reload --port 8001
```

Set `LLM_PROVIDER=ollama` (default), `LLM_PROVIDER=llama`, or `LLM_PROVIDER=rules` in the environment. For Llama, copy `backend/.env.example` to `backend/.env` and set `LLAMA_ADAPTER_PATH` to the trained adapter directory. An explicitly selected Llama provider fails clearly at startup if the adapter, CUDA GPU, or runtime dependencies are unavailable; select Ollama or rules as the fallback provider.

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
| Optional model | 4-bit Llama 3.2 3B + PEFT LoRA adapter |
| Optional retrieval | Local lexical search over JSONL knowledge documents |
| Fallback | Keyword rule engine |
| API | FastAPI |
| Frontend | Vanilla JS / HTML / CSS |

## QLoRA Training and Retrieval

The existing IT categories and response contract stay authoritative. Llama returns the same category, priority, sentiment, summary, action, confidence, and keyword fields consumed by the dashboard. Malformed outputs, model errors, retrieval errors, and confidence below `CLASSIFICATION_CONFIDENCE_THRESHOLD` use the existing rule-based fallback. This project currently has no human-review queue.

The optional local retriever reads one JSON object per line from `RAG_DATA_PATH`, with `title` and either `content` or `text` fields. It ranks documents by lexical overlap and includes up to `RAG_TOP_K` entries. This is a small local retrieval implementation, not a vector database. Ticket and retrieved text are bounded and escaped in the prompt; retrieved documents are explicitly treated as untrusted reference data.

### Data format and training

Create a JSONL file with one labeled ticket per line. Prefer `group_id`, `thread_id`, or `customer_id` so related tickets stay in one split; exact duplicate normalized ticket text is grouped when no identifier is provided.

```json
{"ticket":"VPN authentication fails after a password reset.","context":"VPN policy: authentication failures should be checked with the access team.","answer":{"category":"Access & Permissions","priority":"High","sentiment":"Neutral","summary":"The customer cannot authenticate to the VPN after a password reset.","suggested_action":"Route to the access team and verify the account state.","confidence":0.9,"keywords":["VPN","authentication"]},"group_id":"customer-42"}
```

Use only the application's existing category values shown in the API schema above. For example, this project does not define a separate `BILLING` category.

```bash
python -m backend.training.prepare_dataset tickets.jsonl --output-dir backend/training/data
python -m backend.training.train_qlora --data-dir backend/training/data --output-dir backend/artifacts/llama-ticket-adapter
```

Preparation deterministically assigns distinct ticket/customer/thread groups to train, validation, and test. Training consumes train and validation only; the evaluation script consumes the held-out test file. Use enough independent groups and inspect the split sizes before training. Large datasets with semantic near-duplicates should be grouped upstream by customer/thread/time because text normalization only catches exact normalized duplicates.

LoRA defaults are `LORA_R=32`, `LORA_ALPHA=64`, `LORA_DROPOUT=0.05`, and `TARGET_MODULES=q_proj,k_proj,v_proj,o_proj`. These are starting points, not tuned values. With a sufficiently large, diverse dataset, compare ranks and consider MLP modules such as `gate_proj,up_proj,down_proj`; evaluate each change on the same held-out test set.

### Inference and evaluation

```bash
# From the repository root; prepare backend/.env with LLM_PROVIDER=llama and adapter path.
uvicorn backend.api:app --reload --port 8001
python -m backend.training.evaluate --test backend/training/data/test.jsonl --output backend/training/evaluation-results.json
```

The comparison reports the rule baseline, QLoRA without retrieval, and QLoRA with retrieval: accuracy, macro/per-class precision, recall and F1, confusion matrix, invalid JSON rate, mean latency, Brier score, and expected calibration error. Confidence metrics are diagnostics, not proof of calibration; use a sufficiently large representative test set and calibrate thresholds against actual review outcomes. Results are not included in this repository, and no training or evaluation run is claimed here.

QLoRA teaches task behavior, category boundaries, output structure, and terminology. Retrieval supplies current or changing policies and knowledge; do not train frequently changing company facts into the adapter.

---

## 👤 Author

**Reza (Ray) Aliyari**
[linkedin.com/in/rezaaliyari](https://linkedin.com/in/rezaaliyari)

---

## 📄 License

MIT License
