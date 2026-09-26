"""Small local lexical retriever for optional JSONL knowledge documents."""

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalContextRetriever:
    def __init__(self, documents=None, top_k=5):
        self.documents = documents or []
        self.top_k = max(1, int(top_k))

    @classmethod
    def from_environment(cls):
        path = os.getenv("RAG_DATA_PATH", "").strip()
        top_k = int(os.getenv("RAG_TOP_K", "5"))
        if not path:
            return cls(top_k=top_k)

        documents = []
        try:
            with Path(path).open(encoding="utf-8") as dataset:
                for line in dataset:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content") or item.get("text") or ""
                    if isinstance(content, str) and content.strip():
                        documents.append({
                            "title": str(item.get("title", ""))[:200],
                            "content": content[:4000],
                        })
        except (OSError, ValueError, TypeError) as error:
            logger.warning("Could not load retrieval data (%s); retrieval is disabled.", type(error).__name__)
        return cls(documents, top_k)

    def retrieve(self, query):
        query_terms = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
        if not query_terms:
            return ""
        ranked = []
        for document in self.documents:
            text = f"{document['title']} {document['content']}"
            terms = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
            score = len(query_terms & terms)
            if score:
                ranked.append((score, document))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return "\n\n".join(
            f"{document['title']}\n{document['content']}".strip()
            for _, document in ranked[:self.top_k]
        )