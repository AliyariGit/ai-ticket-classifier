"""Prepare grouped JSONL ticket examples as train/validation/test splits."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from backend.classifier import CATEGORIES, PROMPT_TEMPLATE, PRIORITIES, SENTIMENT_LABELS, TicketClassifier


def normalized_group(record):
    group = record.get("group_id") or record.get("thread_id") or record.get("customer_id")
    if group:
        return str(group)
    normalized = re.sub(r"\W+", " ", record["ticket"].lower()).strip()
    return "ticket-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def render_example(record):
    answer = record["answer"]
    if answer.get("category") not in CATEGORIES:
        raise ValueError(f"Unsupported category: {answer.get('category')!r}")
    if TicketClassifier.parse_and_validate(json.dumps(answer)) is None:
        raise ValueError("Each answer must satisfy the classifier's complete output schema.")
    ticket = record["ticket"]
    context = record.get("context", "")
    prompt = PROMPT_TEMPLATE.format(
        ticket_text=ticket,
        context=context,
        categories=", ".join(CATEGORIES),
        priorities=", ".join(PRIORITIES),
        sentiments=", ".join(SENTIMENT_LABELS),
    )
    return prompt + "\n" + json.dumps(answer, ensure_ascii=True)


def prepare(input_path, output_dir):
    records = []
    with Path(input_path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("ticket"), str) or not record["ticket"].strip():
                raise ValueError(f"Line {line_number} must contain a non-empty ticket.")
            if not isinstance(record.get("answer"), dict):
                raise ValueError(f"Line {line_number} must contain an answer object.")
            records.append((normalized_group(record), {
                "ticket": record["ticket"],
                "context": record.get("context", ""),
                "answer": record["answer"],
                "text": render_example(record),
            }))

    if len(records) < 3:
        raise ValueError("At least three examples are required to create train/validation/test splits.")

    groups = {}
    for group, example in records:
        groups.setdefault(group, []).append(example)
    ordered_groups = sorted(groups, key=lambda key: hashlib.sha256(key.encode("utf-8")).hexdigest())
    if len(ordered_groups) < 3:
        raise ValueError("Need at least 3 distinct ticket/customer/thread groups for non-empty splits.")
    validation_count = max(1, round(len(ordered_groups) * 0.1))
    test_count = max(1, round(len(ordered_groups) * 0.1))
    train_count = len(ordered_groups) - validation_count - test_count
    split_groups = {"train": [], "validation": [], "test": []}
    for index, group in enumerate(ordered_groups):
        split = "train" if index < train_count else "validation" if index < train_count + validation_count else "test"
        split_groups[split].extend(groups[group])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in split_groups.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as target:
            for example in examples:
                target.write(json.dumps(example, ensure_ascii=True) + "\n")
    return {name: len(examples) for name, examples in split_groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Source JSONL with ticket, answer, and optional context/group_id")
    parser.add_argument("--output-dir", default="backend/training/data")
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()