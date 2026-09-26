"""Compare rule baseline, adapter without retrieval, and adapter with retrieval."""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from backend.classifier import CATEGORIES, TicketClassifier
from backend.llama_classifier import LlamaTicketClassifier
from backend.retrieval import LocalContextRetriever


def metrics(rows):
    confusion = {label: {predicted: 0 for predicted in (*CATEGORIES, "INVALID")} for label in CATEGORIES}
    correct = 0
    brier_total = 0.0
    ece_bins = [[0, 0.0, 0.0] for _ in range(10)]
    per_class = {}
    invalid_json = sum(row["prediction"] is None for row in rows)
    for row in rows:
        expected = row["expected"]
        prediction = row["prediction"]
        if prediction not in CATEGORIES:
            confusion[expected]["INVALID"] += 1
            continue
        confusion[expected][prediction] += 1
        is_correct = prediction == expected
        correct += is_correct
        confidence = row["confidence"]
        brier_total += (confidence - float(is_correct)) ** 2
        bucket = min(int(confidence * 10), 9)
        ece_bins[bucket][0] += 1
        ece_bins[bucket][1] += confidence
        ece_bins[bucket][2] += float(is_correct)

    for label in CATEGORIES:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in CATEGORIES if other != label)
        false_negative = sum(confusion[label][other] for other in CATEGORIES if other != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    count = len(rows)
    ece = sum(
        bucket[0] / (count - invalid_json) * abs(bucket[1] / bucket[0] - bucket[2] / bucket[0])
        for bucket in ece_bins if bucket[0]
    ) if count > invalid_json else 0
    macro = {
        name: sum(value[name] for value in per_class.values()) / len(CATEGORIES)
        for name in ("precision", "recall", "f1")
    }
    return {
        "count": count,
        "accuracy": correct / count if count else 0,
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "per_class": per_class,
        "confusion_matrix": confusion,
        "invalid_json_rate": invalid_json / count if count else 0,
        "mean_latency_seconds": sum(row["latency"] for row in rows) / count if count else 0,
        "confidence_brier_score": brier_total / (count - invalid_json) if count > invalid_json else None,
        "confidence_expected_calibration_error": ece,
    }


def evaluate(test_path, output_path):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    with Path(test_path).open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    adapter = LlamaTicketClassifier(
        os.getenv("LLAMA_BASE_MODEL", "meta-llama/Llama-3.2-3B"),
        os.environ["LLAMA_ADAPTER_PATH"],
    )
    classifier = TicketClassifier(provider="rules", use_llm=False)
    retriever = LocalContextRetriever.from_environment()
    outputs = {"baseline": [], "qlora": [], "qlora_rag": []}
    for record in records:
        ticket = record["ticket"]
        expected = record["answer"]["category"]
        started = time.perf_counter()
        baseline = classifier.classify_rule_based(ticket)
        outputs["baseline"].append({
            "expected": expected, "prediction": baseline["category"],
            "confidence": baseline["confidence"], "latency": time.perf_counter() - started,
        })
        for name in ("qlora", "qlora_rag"):
            started = time.perf_counter()
            context = retriever.retrieve(ticket) if name == "qlora_rag" else ""
            raw = adapter.generate(classifier.build_prompt(ticket, context))
            prediction = classifier.parse_and_validate(raw)
            outputs[name].append({
                "expected": expected,
                "prediction": prediction["category"] if prediction else None,
                "confidence": prediction["confidence"] if prediction else 0.0,
                "latency": time.perf_counter() - started,
            })
    report = {name: metrics(rows) for name, rows in outputs.items()}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="backend/training/data/test.jsonl")
    parser.add_argument("--output", default="backend/training/evaluation-results.json")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.test, args.output), indent=2))


if __name__ == "__main__":
    main()