"""
AI Ticket Classifier API
Author: Reza (Ray) Aliyari
Description: REST API for ticket classification with analytics
"""

import json
from datetime import datetime
from collections import Counter
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from classifier import TicketClassifier

app = FastAPI(
    title="AI Ticket Classifier API",
    description="NLM-based IT support ticket classification and analytics",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = TicketClassifier()

# In-memory store for demo (use DB in production)
classified_tickets: List[dict] = []

# Seed with demo data
DEMO_TICKETS = [
    {"id": "TKT-001", "text": "VPN not working, can't connect to office network. Authentication keeps failing."},
    {"id": "TKT-002", "text": "Excel crashes every time I open a large spreadsheet. Tried reinstalling but same issue."},
    {"id": "TKT-003", "text": "Please add dark mode to the internal portal. Would be great for late-night work sessions!"},
    {"id": "TKT-004", "text": "URGENT: Production database is down! All orders are failing. Need immediate help!"},
    {"id": "TKT-005", "text": "My mouse is double-clicking when I only click once. Tried different USB port, same problem."},
    {"id": "TKT-006", "text": "Suspicious email received claiming to be from IT asking for my password. What should I do?"},
    {"id": "TKT-007", "text": "System has been very slow since the last update. Takes 5 minutes to boot."},
    {"id": "TKT-008", "text": "Cannot access the HR system. Getting Access Denied error since yesterday."},
]


@app.on_event("startup")
async def seed_demo_data():
    """Pre-classify demo tickets on startup."""
    for ticket in DEMO_TICKETS:
        result = classifier.classify(ticket["text"], ticket["id"])
        classified_tickets.append(result)


class TicketRequest(BaseModel):
    text: str
    ticket_id: Optional[str] = None


class BatchRequest(BaseModel):
    tickets: List[TicketRequest]


@app.get("/")
def root():
    return {"message": "AI Ticket Classifier API", "version": "1.0.0", "docs": "/docs"}


@app.post("/classify")
def classify_ticket(req: TicketRequest):
    """Classify a single support ticket."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Ticket text cannot be empty.")
    try:
        result = classifier.classify(req.text, req.ticket_id)
        classified_tickets.append(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch")
def classify_batch(req: BatchRequest):
    """Classify multiple tickets at once."""
    tickets = [{"text": t.text, "id": t.ticket_id} for t in req.tickets]
    results = classifier.batch_classify(tickets)
    classified_tickets.extend(results)
    return {"count": len(results), "results": results}


@app.get("/tickets")
def get_tickets(limit: int = 50, priority: Optional[str] = None, category: Optional[str] = None):
    """Get all classified tickets with optional filtering."""
    results = classified_tickets[-limit:]
    if priority:
        results = [t for t in results if t.get("priority") == priority]
    if category:
        results = [t for t in results if t.get("category") == category]
    return {"count": len(results), "tickets": results}


@app.get("/analytics")
def get_analytics():
    """Get aggregate analytics across all classified tickets."""
    if not classified_tickets:
        return {"message": "No tickets classified yet."}

    categories = Counter(t.get("category") for t in classified_tickets)
    priorities = Counter(t.get("priority") for t in classified_tickets)
    sentiments = Counter(t.get("sentiment") for t in classified_tickets)
    methods = Counter(t.get("method") for t in classified_tickets)

    avg_confidence = sum(t.get("confidence", 0) for t in classified_tickets) / len(classified_tickets)

    return {
        "total_tickets": len(classified_tickets),
        "avg_confidence": round(avg_confidence, 3),
        "by_category": dict(categories.most_common()),
        "by_priority": dict(priorities),
        "by_sentiment": dict(sentiments),
        "by_method": dict(methods),
        "critical_tickets": [
            t for t in classified_tickets if t.get("priority") == "Critical"
        ],
    }


@app.delete("/tickets/reset")
def reset_tickets():
    """Clear all tickets (except demo data)."""
    classified_tickets.clear()
    return {"message": "Tickets cleared."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
