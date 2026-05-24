"""
AI Ticket Classifier
Author: Reza (Ray) Aliyari
Description: NLM-based IT support ticket classification using LLMs and traditional ML
"""

import json
import re
import logging
from typing import Optional
from datetime import datetime

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ticket categories and priorities
CATEGORIES = [
    "Hardware Issue",
    "Software Bug",
    "Network & Connectivity",
    "Access & Permissions",
    "Performance Issue",
    "Security Incident",
    "Data Loss / Backup",
    "Feature Request",
    "General Inquiry",
    "Other",
]

PRIORITIES = ["Critical", "High", "Medium", "Low"]

SENTIMENT_LABELS = ["Frustrated", "Neutral", "Satisfied"]

PROMPT_TEMPLATE = """You are an expert IT support ticket classifier. Analyze the following ticket and return a JSON object with classification.

Ticket:
---
{ticket_text}
---

Respond ONLY with a valid JSON object in this exact format:
{{
  "category": "<one of: {categories}>",
  "priority": "<one of: Critical, High, Medium, Low>",
  "sentiment": "<one of: Frustrated, Neutral, Satisfied>",
  "summary": "<one sentence summary of the issue>",
  "suggested_action": "<brief recommended next step for the support team>",
  "confidence": <float between 0.0 and 1.0>,
  "keywords": ["<keyword1>", "<keyword2>", "<keyword3>"]
}}

Rules:
- Critical: system down, security breach, data loss
- High: major feature broken, affects many users
- Medium: partial functionality broken, workaround exists
- Low: minor issue, cosmetic, general question
- Confidence reflects how clearly the ticket maps to the category
"""


class TicketClassifier:
    """
    Classifies IT support tickets using local LLMs (via Ollama) or
    falls back to rule-based classification.
    """

    def __init__(
        self,
        model: str = "phi3",
        ollama_url: str = "http://localhost:11434",
        use_llm: bool = True,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.use_llm = use_llm
        self._check_ollama()

    def _check_ollama(self):
        """Check if Ollama is available."""
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if r.status_code == 200:
                logger.info(f"Ollama connected. Model: {self.model}")
            else:
                logger.warning("Ollama not responding — falling back to rule-based.")
                self.use_llm = False
        except Exception:
            logger.warning("Ollama not found — using rule-based classifier.")
            self.use_llm = False

    def classify(self, ticket_text: str, ticket_id: Optional[str] = None) -> dict:
        """
        Classify a single ticket. Returns full classification result.
        """
        if not ticket_text.strip():
            raise ValueError("Ticket text cannot be empty.")

        if self.use_llm:
            result = self._classify_with_llm(ticket_text)
        else:
            result = self._classify_rule_based(ticket_text)

        result["ticket_id"] = ticket_id or f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        result["classified_at"] = datetime.now().isoformat()
        result["ticket_text"] = ticket_text[:300]
        result["method"] = "LLM" if self.use_llm else "Rule-Based"

        return result

    def _classify_with_llm(self, ticket_text: str) -> dict:
        """Use Ollama LLM for classification."""
        prompt = PROMPT_TEMPLATE.format(
            ticket_text=ticket_text[:1500],
            categories=", ".join(CATEGORIES),
        )

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=60,
            )
            response.raise_for_status()
            raw = response.json().get("response", "")

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                logger.warning("LLM response malformed — falling back to rules.")
                return self._classify_rule_based(ticket_text)

        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return self._classify_rule_based(ticket_text)

    def _classify_rule_based(self, text: str) -> dict:
        """Simple keyword-based classification fallback."""
        text_lower = text.lower()

        # Category rules
        category = "General Inquiry"
        if any(w in text_lower for w in ["crash", "blue screen", "bsod", "hardware", "keyboard", "mouse", "monitor", "printer"]):
            category = "Hardware Issue"
        elif any(w in text_lower for w in ["error", "bug", "crash", "software", "application", "app", "program"]):
            category = "Software Bug"
        elif any(w in text_lower for w in ["network", "internet", "vpn", "wifi", "connection", "timeout"]):
            category = "Network & Connectivity"
        elif any(w in text_lower for w in ["access", "permission", "password", "login", "locked", "unauthorized"]):
            category = "Access & Permissions"
        elif any(w in text_lower for w in ["slow", "performance", "lag", "freeze", "hang", "unresponsive"]):
            category = "Performance Issue"
        elif any(w in text_lower for w in ["security", "breach", "hack", "virus", "malware", "phishing"]):
            category = "Security Incident"
        elif any(w in text_lower for w in ["data", "backup", "lost", "deleted", "recovery", "restore"]):
            category = "Data Loss / Backup"
        elif any(w in text_lower for w in ["feature", "request", "enhancement", "improvement", "add"]):
            category = "Feature Request"

        # Priority rules
        priority = "Medium"
        if any(w in text_lower for w in ["urgent", "critical", "down", "outage", "cannot work", "production"]):
            priority = "Critical"
        elif any(w in text_lower for w in ["broken", "cannot", "unable", "failing", "not working"]):
            priority = "High"
        elif any(w in text_lower for w in ["question", "how to", "feature request", "inquiry"]):
            priority = "Low"

        # Sentiment rules
        sentiment = "Neutral"
        if any(w in text_lower for w in ["frustrated", "angry", "ridiculous", "unacceptable", "terrible", "worst"]):
            sentiment = "Frustrated"
        elif any(w in text_lower for w in ["thank", "please", "appreciate", "great"]):
            sentiment = "Satisfied"

        keywords = [w for w in text_lower.split() if len(w) > 5][:3]

        return {
            "category": category,
            "priority": priority,
            "sentiment": sentiment,
            "summary": f"{category} reported by user.",
            "suggested_action": f"Route to {category} team and respond within SLA.",
            "confidence": 0.65,
            "keywords": keywords,
        }

    def batch_classify(self, tickets: list[dict]) -> list[dict]:
        """Classify a batch of tickets. Each ticket dict must have 'text' field."""
        results = []
        for i, ticket in enumerate(tickets):
            logger.info(f"Classifying ticket {i+1}/{len(tickets)}...")
            result = self.classify(
                ticket.get("text", ""),
                ticket.get("id"),
            )
            results.append(result)
        return results


if __name__ == "__main__":
    classifier = TicketClassifier()

    # Test ticket
    test_ticket = """
    Hi, I've been unable to access the VPN since this morning. 
    I keep getting an error: 'Authentication failed - server unreachable'.
    This is blocking my entire team from working remotely. 
    We have a client presentation in 2 hours and this is absolutely critical.
    """

    result = classifier.classify(test_ticket, ticket_id="TKT-001")
    print("\n=== Classification Result ===")
    print(json.dumps(result, indent=2))
