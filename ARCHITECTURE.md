# How It Works — End-to-End Architecture

This document walks through every layer of the AI Ticket Classifier: from a user typing a ticket in the browser, through FastAPI, into the classification engine, and back to the dashboard.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Component Map](#2-component-map)
3. [Server Startup Sequence](#3-server-startup-sequence)
4. [End-to-End: Classifying a Ticket](#4-end-to-end-classifying-a-ticket)
5. [FastAPI — How It Receives and Routes Requests](#5-fastapi--how-it-receives-and-routes-requests)
6. [Classification Engine — LLM Path](#6-classification-engine--llm-path)
7. [Classification Engine — Rule-Based Fallback](#7-classification-engine--rule-based-fallback)
8. [Frontend — How It Fetches Data](#8-frontend--how-it-fetches-data)
9. [Analytics Endpoint — Aggregating Results](#9-analytics-endpoint--aggregating-results)
10. [Data Schema Reference](#10-data-schema-reference)
11. [Error Handling & Fallback Strategy](#11-error-handling--fallback-strategy)

---

## 1. System Overview

The application has three layers:

```
┌──────────────────────────────────────────────┐
│              BROWSER (port 3001)             │
│         index.html — Vanilla JS SPA          │
│  User submits ticket → fetch() → REST API    │
└──────────────────────┬───────────────────────┘
                       │  HTTP (JSON)
                       ▼
┌──────────────────────────────────────────────┐
│           FASTAPI SERVER (port 8001)         │
│  api.py — Routes, validation, in-memory DB   │
│  Receives request → calls classifier         │
└──────────────────────┬───────────────────────┘
                       │  Python function call
                       ▼
┌──────────────────────────────────────────────┐
│          CLASSIFICATION ENGINE               │
│  classifier.py — provider, prompt, parsing,  │
│  validation, confidence fallback              │
│  retrieval.py — optional local JSONL search  │
│  llama_classifier.py — 4-bit base + PEFT     │
│   ┌───────────┐ ┌──────────────┐ ┌─────────┐ │
│   │ Ollama    │ │ QLoRA Llama  │ │ Rules   │ │
│   │ Phi-3     │ │ 3.2 3B       │ │ fallback│ │
│   └───────────┘ └──────────────┘ └─────────┘ │
└──────────────────────────────────────────────┘
```

**Ports at a glance:**

| Service | Port | Protocol |
|---------|------|----------|
| Frontend (static files) | 3001 | HTTP |
| FastAPI backend | 8001 | HTTP/REST |
| Ollama LLM server | 11434 | HTTP |

---

## 2. Component Map

```
ai-ticket-classifier/
│
├── backend/
│   ├── api.py           ← FastAPI app: routes, request models, CORS, in-memory store
│   ├── classifier.py    ← provider selection, retrieval, prompt and output validation
│   ├── retrieval.py     ← optional JSONL lexical retriever
│   ├── llama_classifier.py ← 4-bit Llama inference + separately loaded PEFT adapter
│   ├── training/        ← dataset preparation, QLoRA training and evaluation
│   ├── tests/           ← GPU-free classifier and retrieval tests
│   ├── requirements.txt ← base Python dependencies
│   └── requirements-llama.txt ← optional Hugging Face/QLoRA dependencies
│
└── frontend/
    └── index.html       ← Single-page app: dashboard, charts, classification form
```

**Dependency chain:**

```
index.html
  └── fetch("http://localhost:8001/classify")
        └── api.py → POST /classify
              └── classifier.classify(text)
              ├── retrieval.py → relevant local JSONL context (optional)
              ├── Ollama provider → HTTP POST (localhost:11434)
              ├── Llama provider → 4-bit base + separate LoRA adapter
              ├── validate category, fields, confidence and output bounds
              └── keyword rules on errors or low confidence
```

---

## 3. Server Startup Sequence

When you run `uvicorn api:app --port 8001` from `backend/`, here is what happens in order:

```
Step 1 — FastAPI app object created (api.py line 16)
          app = FastAPI(title="AI Ticket Classifier API", ...)

Step 2 — CORS middleware registered (api.py lines 22–28)
          Allows the frontend (any origin) to call the API

Step 3 — Environment loaded and TicketClassifier instantiated
          LLM_PROVIDER selects ollama (default), llama, or rules
          CLASSIFICATION_CONFIDENCE_THRESHOLD defaults to 0.80
          RAG_DATA_PATH optionally loads local JSONL reference documents

Step 4 — Provider initialization
          ollama → GET /api/tags; unavailable Ollama selects rule fallback
          llama  → require adapter path, CUDA GPU and optional ML packages;
                   load 4-bit base model and separate frozen PEFT adapter
          rules  → skip model initialization

Step 5 — @app.on_event("startup") seeds 8 demo tickets
          Each ticket passes through retrieval, the selected provider,
          output validation and confidence fallback; results are stored
          in the existing in-memory list.

Step 6 — Uvicorn listens on 0.0.0.0:8001
          Explicitly selecting llama without its runtime requirements
          raises a startup error instead of attempting CPU quantized load.
```

After startup the in-memory list already contains 8 pre-classified demo tickets, so the dashboard has data to show immediately.

---

## 4. End-to-End: Classifying a Ticket

This is the complete journey of a single ticket from browser to response.

```
[Browser]
User types: "VPN is down, team cannot work. Urgent!"
User clicks: "Classify Ticket" button

    │
    │  JavaScript (index.html)
    │  classifyTicket() function called
    │  fetch("http://localhost:8001/classify", {
    │    method: "POST",
    │    headers: { "Content-Type": "application/json" },
    │    body: JSON.stringify({ text: "VPN is down..." })
    │  })
    │
    ▼

[FastAPI — api.py]
POST /classify route receives the request

    │
    │  Pydantic parses request body into TicketRequest:
    │    TicketRequest(text="VPN is down...", ticket_id=None)
    │
    │  Validation: text.strip() must not be empty
    │
    │  Calls: result = classifier.classify(req.text, req.ticket_id)
    │
    ▼

[TicketClassifier — classifier.py]
classify() method:

    │
     │  Retrieve up to RAG_TOP_K matching JSONL documents (optional)
     │  Treat ticket and context as untrusted data in separate prompt blocks
     │
     ├─ LLM_PROVIDER=ollama ─→ HTTP POST to Ollama (Phi-3 by default)
     ├─ LLM_PROVIDER=llama  ─→ 4-bit Llama 3.2 3B + separate PEFT adapter
     └─ LLM_PROVIDER=rules  ─→ keyword rules
     │
     │  Parse bounded JSON and validate all existing enum fields,
     │  confidence range, string lengths and keyword count
     │
     └─ Invalid output, provider error, retrieval failure or confidence
         below threshold → continue without context or use rule fallback
    │
    │  Enrich result (both paths):
    │    result["ticket_id"]    = "TKT-20260611103000"
    │    result["classified_at"] = "2026-06-11T10:30:00"
    │    result["ticket_text"]  = first 300 chars
    │    result["method"]       = "LLM", "QLoRA Llama" or "Rule-Based"
    │
    ▼

[FastAPI — api.py, back in classify_ticket()]

    │  classified_tickets.append(result)   ← stored in memory
    │  return result                        ← serialised as JSON
    │
    ▼

[HTTP Response — 200 OK]
{
  "ticket_id": "TKT-20260611103000",
  "category": "Network & Connectivity",
  "priority": "Critical",
  "sentiment": "Frustrated",
  "summary": "VPN authentication failing, blocking remote team access.",
  "suggested_action": "Escalate to network team immediately.",
  "confidence": 0.94,
  "keywords": ["vpn", "network", "urgent"],
  "ticket_text": "VPN is down, team cannot work. Urgent!",
  "method": "LLM",
  "classified_at": "2026-06-11T10:30:00"
}

    │
    ▼

[Browser — index.html]
fetch() promise resolves with the JSON above
classifyTicket() renders a result card
loadData() called → refreshes ticket table and analytics charts
```

---

## 5. FastAPI — How It Receives and Routes Requests

FastAPI is a Python web framework that maps HTTP routes to Python functions. Here is how each route is defined and what it does:

### Route definitions (api.py)

```python
@app.post("/classify")
def classify_ticket(req: TicketRequest):
```

The `@app.post("/classify")` decorator tells FastAPI: when a POST request arrives at `/classify`, call this function.

FastAPI automatically:
- Reads the JSON body from the HTTP request
- Validates it against the `TicketRequest` Pydantic model
- Injects the validated object as the `req` parameter
- Serializes the return value back to JSON
- Sets the correct `Content-Type: application/json` header

### All routes at a glance

| Route | Method | What it does |
|-------|--------|--------------|
| `/` | GET | Health check, returns version info |
| `/classify` | POST | Classify one ticket, store in memory, return result |
| `/classify/batch` | POST | Loop over a list of tickets, classify each, return all |
| `/tickets` | GET | Return stored tickets, supports `?limit=`, `?priority=`, `?category=` filters |
| `/analytics` | GET | Compute category/priority/sentiment counts across all tickets |
| `/tickets/reset` | DELETE | Clear `classified_tickets` list |

### CORS middleware

The frontend runs on port 3001; the API runs on 8001. Browsers block cross-origin requests by default. The CORS middleware in `api.py` adds the `Access-Control-Allow-Origin: *` header to every response, which allows the frontend to call the API freely:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins
    allow_methods=["*"],   # allow GET, POST, DELETE, etc.
    allow_headers=["*"],
)
```

### Pydantic models (request validation)

```python
class TicketRequest(BaseModel):
    text: str
    ticket_id: Optional[str] = None
```

FastAPI uses this model to parse and validate the JSON body. If `text` is missing from the request body, FastAPI returns a `422 Unprocessable Entity` error automatically before the route function is called.

---

## 6. Classification Engine — LLM Path

When Ollama or the Llama provider is selected, the classifier builds the same task prompt and validates the model output:

```
1. Build the prompt
   ─────────────────
   PROMPT_TEMPLATE is filled with:
    - Ticket text (truncated to 1500 characters and escaped as untrusted input)
    - Up to 6000 characters of retrieved local reference context
    - Existing category, priority, and sentiment enums

   The prompt instructs the model to return ONLY a JSON object with
   these fields: category, priority, sentiment, summary,
   suggested_action, confidence, keywords

2. Generate a response
     ────────────────────
     Ollama → POST /api/generate, default model `phi3`, timeout 60 seconds
     Llama  → CUDA inference with a 4-bit NF4 base and separately loaded
                        frozen PEFT adapter; at most 300 new tokens

3. Parse the response
   ──────────────────
   Ollama returns:
   { "response": "...JSON object here..." }

     The parser accepts one bounded JSON object, then validates the existing
     category, priority, and sentiment enums; confidence must be finite and
     between 0 and 1; text and keyword lengths are capped.

4. Fallback and confidence gate
    ────────────────────────────
    Retrieval failure means empty context. Provider errors, malformed or
    invalid output, and confidence below CLASSIFICATION_CONFIDENCE_THRESHOLD
    use the existing keyword-rule classifier. The default threshold is 0.80.
```

**Why Phi-3?**
Phi-3 is a small (3.8B parameter) Microsoft model that runs well on CPU. It follows instruction prompts reliably and returns structured JSON consistently, making it well-suited for this classification task without requiring a GPU.
The Llama provider requires a supported CUDA GPU and an adapter directory. It fails clearly during application initialization when those requirements are unavailable; it does not silently attempt CPU quantized inference. Select Ollama or rules explicitly for a fallback provider.

---

## 7. Classification Engine — Rule-Based Fallback

When Ollama is not available, a model request fails, output is invalid, or confidence is below threshold, `classify_rule_based()` applies keyword matching:

```
Input: ticket text (lowercased)

Category detection (first match wins):
┌────────────────────────┬──────────────────────────────────────────────┐
│ Category               │ Trigger keywords                             │
├────────────────────────┼──────────────────────────────────────────────┤
│ Hardware Issue         │ crash, blue screen, bsod, keyboard, mouse... │
│ Software Bug           │ error, bug, crash, application, program...   │
│ Network & Connectivity │ network, internet, vpn, wifi, connection...  │
│ Access & Permissions   │ access, permission, password, login...       │
│ Performance Issue      │ slow, lag, freeze, hang, unresponsive...     │
│ Security Incident      │ security, breach, hack, virus, phishing...   │
│ Data Loss / Backup     │ data, backup, lost, deleted, recovery...     │
│ Feature Request        │ feature, request, enhancement, add...        │
│ General Inquiry        │ (default if no match)                        │
└────────────────────────┴──────────────────────────────────────────────┘

Priority detection:
┌──────────┬────────────────────────────────────────────────┐
│ Priority │ Trigger keywords                               │
├──────────┼────────────────────────────────────────────────┤
│ Critical │ urgent, critical, down, outage, production...  │
│ High     │ broken, cannot, unable, failing, not working   │
│ Low      │ question, how to, feature request, inquiry     │
│ Medium   │ (default)                                      │
└──────────┴────────────────────────────────────────────────┘

Sentiment detection:
┌───────────┬──────────────────────────────────────────────┐
│ Sentiment │ Trigger keywords                             │
├───────────┼──────────────────────────────────────────────┤
│ Frustrated│ frustrated, angry, ridiculous, terrible...   │
│ Satisfied │ thank, please, appreciate, great             │
│ Neutral   │ (default)                                    │
└───────────┴──────────────────────────────────────────────┘

Output: dict with confidence fixed at 0.65
```

The rule-based path never makes network calls — it is pure Python string operations, so it always succeeds instantly.

---

## 8. Frontend — How It Fetches Data

`index.html` is a single-page application with no build step or framework. All API calls use the browser's native `fetch()` API.

### On page load

```javascript
// Called once when the page finishes loading
document.addEventListener("DOMContentLoaded", () => {
    loadData();                          // initial data fetch
    setInterval(loadData, 30000);        // auto-refresh every 30 seconds
});
```

### loadData() — the main fetch sequence

```javascript
async function loadData() {
    // 1. Fetch all tickets (last 50)
    const ticketsRes = await fetch("http://localhost:8001/tickets?limit=50");
    const ticketsData = await ticketsRes.json();
    allTickets = ticketsData.tickets;

    // 2. Fetch analytics summary
    const analyticsRes = await fetch("http://localhost:8001/analytics");
    analyticsData = await analyticsRes.json();

    // 3. Re-render the entire UI
    renderAll();
}
```

Both fetches run sequentially (analytics depends on having fresh ticket data in context). The results update the module-level variables `allTickets` and `analyticsData`.

### classifyTicket() — submitting a new ticket

```javascript
async function classifyTicket() {
    const text = document.getElementById("ticketText").value.trim();
    if (!text) return;

    // Show spinner, disable button
    setLoading(true);

    const res = await fetch("http://localhost:8001/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    });

    const result = await res.json();

    // Render the result card above the form
    renderResult(result);

    // Refresh all tables and charts
    await loadData();

    setLoading(false);
}
```

### renderCharts() — visualising analytics

Charts are plain HTML elements rendered by JavaScript — no chart library required. Each bar is a `<div>` with a width set as a percentage of the maximum count:

```javascript
function renderBar(label, count, maxCount, color) {
    const pct = maxCount > 0 ? (count / maxCount) * 100 : 0;
    return `
        <div class="bar-row">
            <span class="bar-label">${label}</span>
            <div class="bar-track">
                <div class="bar-fill" style="width:${pct}%; background:${color}"></div>
            </div>
            <span class="bar-count">${count}</span>
        </div>`;
}
```

---

## 9. Analytics Endpoint — Aggregating Results

`GET /analytics` computes statistics on-the-fly from the in-memory list:

```python
@app.get("/analytics")
def get_analytics():
    categories = Counter(t.get("category") for t in classified_tickets)
    priorities = Counter(t.get("priority") for t in classified_tickets)
    sentiments = Counter(t.get("sentiment") for t in classified_tickets)
    methods    = Counter(t.get("method")   for t in classified_tickets)

    avg_confidence = sum(t.get("confidence", 0) for t in classified_tickets) \
                     / len(classified_tickets)

    return {
        "total_tickets":    len(classified_tickets),
        "avg_confidence":   round(avg_confidence, 3),
        "by_category":      dict(categories.most_common()),
        "by_priority":      dict(priorities),
        "by_sentiment":     dict(sentiments),
        "by_method":        dict(methods),
        "critical_tickets": [t for t in classified_tickets
                             if t.get("priority") == "Critical"],
    }
```

`Counter` from Python's `collections` module counts occurrences by key in a single pass. The result is computed fresh on every request — there is no caching layer.

---

## 10. Data Schema Reference

### Request — POST /classify

```json
{
  "text": "Description of the IT issue (required)",
  "ticket_id": "Optional custom ID string"
}
```

### Response — classification result

```json
{
  "ticket_id":       "TKT-20260611103000",
  "category":        "Network & Connectivity",
  "priority":        "Critical",
  "sentiment":       "Frustrated",
  "summary":         "One sentence describing the issue",
  "suggested_action":"Recommended next step for the support team",
  "confidence":      0.94,
  "keywords":        ["vpn", "network", "urgent"],
  "ticket_text":     "First 300 characters of original input",
  "method":          "LLM",
  "classified_at":   "2026-06-11T10:30:00.123456"
}
```

### Valid enum values

| Field | Values |
|-------|--------|
| `category` | `Hardware Issue`, `Software Bug`, `Network & Connectivity`, `Access & Permissions`, `Performance Issue`, `Security Incident`, `Data Loss / Backup`, `Feature Request`, `General Inquiry`, `Other` |
| `priority` | `Critical`, `High`, `Medium`, `Low` |
| `sentiment` | `Frustrated`, `Neutral`, `Satisfied` |
| `method` | `LLM`, `QLoRA Llama`, `Rule-Based` |

### Response — GET /analytics

```json
{
  "total_tickets":    10,
  "avg_confidence":   0.812,
  "by_category":      { "Network & Connectivity": 3, "Software Bug": 2, ... },
  "by_priority":      { "Critical": 2, "High": 3, "Medium": 4, "Low": 1 },
  "by_sentiment":     { "Frustrated": 4, "Neutral": 5, "Satisfied": 1 },
  "by_method":        { "LLM": 8, "Rule-Based": 2 },
  "critical_tickets": [ ...full ticket objects with priority=Critical... ]
}
```

---

## 11. Error Handling & Fallback Strategy

The system keeps the existing rule-based fallback and adds retrieval and model-output guards:

### Provider initialization

If Ollama is unavailable during startup, classification uses the rules provider. If Llama is explicitly selected but CUDA, the adapter, or optional dependencies are unavailable, initialization raises a clear error instead of silently loading on CPU.

### Per-request behavior

Retrieval exceptions continue with empty context. Model errors, malformed JSON, invalid enum values, invalid confidence, overlong output, or confidence below the configured threshold use the rule-based classifier. The API response shape remains unchanged.

Ticket text and retrieved documents are bounded and escaped before being placed in separate prompt data blocks. The prompt directs the model to treat both as untrusted reference data. This is defense in depth, not a guarantee that prompt injection is impossible.

### FastAPI error responses

| Condition | HTTP Status | Detail |
|-----------|-------------|--------|
| Empty ticket text | 400 | `"Ticket text cannot be empty."` |
| Unexpected exception in classifier | 500 | Exception message |
| Missing required field in request body | 422 | Pydantic validation error detail |

---

*For setup instructions see [README.md](README.md).*
