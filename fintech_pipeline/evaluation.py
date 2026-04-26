from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .metrics import export_evaluation_metrics
from .rag import RAGPipeline
from .utils import read_json, write_json


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    truth_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in truth_tokens:
        truth_counts[token] = truth_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, truth_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


class RAGEvaluator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.rag = RAGPipeline(config)

    def evaluate(self, sample_limit: int | None = None) -> dict[str, Any]:
        dataset_path = Path(self.config["paths"]["dataset_dir"]) / f"{self.config['dataset']['name']}.json"
        dataset = read_json(dataset_path)
        if not dataset:
            raise ValueError(f"No dataset records found in {dataset_path}")

        if sample_limit:
            dataset = dataset[:sample_limit]

        results = []
        exact_matches = 0
        total_f1 = 0.0
        retrieval_hits = 0

        for row in dataset:
            response = self.rag.ask(row["question"])
            predicted = response["answer"]
            sources = response["sources"]

            normalized_pred = normalize_text(predicted)
            normalized_gt = normalize_text(row["answer"])
            em = 1 if normalized_pred == normalized_gt else 0
            f1 = token_f1(predicted, row["answer"])
            hit = 1 if any(source["id"] == row["id"] for source in sources) else 0

            exact_matches += em
            total_f1 += f1
            retrieval_hits += hit
            results.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "ground_truth": row["answer"],
                    "prediction": predicted,
                    "exact_match": em,
                    "f1": round(f1, 4),
                    "retrieval_hit_at_k": hit,
                }
            )

        total = max(1, len(results))
        summary = {
            "samples_evaluated": len(results),
            "exact_match": round(exact_matches / total, 4),
            "token_f1": round(total_f1 / total, 4),
            "retrieval_hit_rate_at_k": round(retrieval_hits / total, 4),
            "top_k": self.config["rag"].get("top_k", 5),
        }

        output_path = Path(self.config["paths"]["rag_dir"]) / "evaluation.json"
        payload = {"summary": summary, "results": results}
        write_json(output_path, payload)
        metrics_dir = Path(self.config["paths"].get("metrics_dir", Path(self.config["paths"]["dataset_dir"]).parent / "metrics"))
        metrics_export = export_evaluation_metrics(metrics_dir, payload)
        return {"summary": summary, "output_file": str(output_path), "metrics": metrics_export}
