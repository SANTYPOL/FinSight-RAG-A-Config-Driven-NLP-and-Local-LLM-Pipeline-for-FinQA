from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .ollama_client import OllamaClient
from .utils import ensure_dir, read_json, write_json

RAG_SYSTEM_PROMPT = """You are a finance RAG assistant.
Answer only from the retrieved context.
If the answer is not supported by the context, reply exactly: Not available in data.
Keep the answer concise and factual.
"""


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query) + 1e-12
    matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-12
    return (matrix @ query) / (matrix_norm * query_norm)


class RAGPipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.client = OllamaClient(
            base_url=config["ollama"]["base_url"],
            timeout=config["ollama"].get("timeout", 120),
        )

    def build_index(self) -> dict[str, Any]:
        dataset_path = Path(self.config["paths"]["dataset_dir"]) / f"{self.config['dataset']['name']}.json"
        dataset = read_json(dataset_path)
        if not dataset:
            raise ValueError(f"No dataset records found in {dataset_path}")

        documents = []
        texts = []
        for row in dataset:
            doc = {
                "id": row["id"],
                "symbol": row["symbol"],
                "question": row["question"],
                "answer": row["answer"],
                "category": row.get("category", "Other"),
                "difficulty": row.get("difficulty", "Medium"),
                "context_hint": row.get("context_hint", ""),
                "source_keywords": row.get("source_keywords", []),
                "source_entities": row.get("source_entities", {}),
                "source_section": row.get("source_section", ""),
                "source_pages": row.get("source_pages", ""),
                "source_file": row.get("source_file", ""),
            }
            documents.append(doc)
            texts.append(
                "\n".join(
                    [
                        f"Symbol: {doc['symbol']}",
                        f"Category: {doc['category']}",
                        f"Question: {doc['question']}",
                        f"Answer: {doc['answer']}",
                        f"Keywords: {', '.join(doc.get('source_keywords', []))}",
                        f"Entities: {doc.get('source_entities', {})}",
                        f"Section: {doc['source_section']}",
                    ]
                )
            )

        embeddings = self.client.embed(self.config["ollama"]["embedding_model"], texts)
        matrix = np.array(embeddings, dtype=np.float32)

        index_dir = ensure_dir(self.config["paths"]["rag_dir"])
        np.save(index_dir / "embeddings.npy", matrix)
        write_json(index_dir / "documents.json", documents)

        metadata = {
            "total_documents": len(documents),
            "embedding_dim": int(matrix.shape[1]),
            "embedding_model": self.config["ollama"]["embedding_model"],
            "dataset_name": self.config["dataset"]["name"],
        }
        write_json(index_dir / "index_metadata.json", metadata)
        return metadata

    def ask(self, query: str) -> dict[str, Any]:
        index_dir = Path(self.config["paths"]["rag_dir"])
        embeddings_path = index_dir / "embeddings.npy"
        documents_path = index_dir / "documents.json"
        if not embeddings_path.exists() or not documents_path.exists():
            raise FileNotFoundError("RAG index is missing. Run the build-rag step first.")

        matrix = np.load(embeddings_path)
        documents = read_json(documents_path)
        query_vector = np.array(
            self.client.embed(self.config["ollama"]["embedding_model"], [query])[0],
            dtype=np.float32,
        )

        scores = cosine_similarity(query_vector, matrix)
        top_k = min(self.config["rag"].get("top_k", 5), len(documents))
        best_indices = np.argsort(scores)[::-1][:top_k]
        retrieved = []
        for index in best_indices:
            document = dict(documents[int(index)])
            document["score"] = round(float(scores[int(index)]), 4)
            retrieved.append(document)

        context = "\n\n".join(
            [
                "\n".join(
                    [
                        f"Symbol: {doc['symbol']}",
                        f"Category: {doc['category']}",
                        f"Question: {doc['question']}",
                        f"Answer: {doc['answer']}",
                        f"Keywords: {', '.join(doc.get('source_keywords', []))}",
                        f"Section: {doc['source_section']}",
                        f"Pages: {doc['source_pages']}",
                    ]
                )
                for doc in retrieved
            ]
        )
        prompt = f"Context:\n{context}\n\nUser question: {query}\n\nAnswer:"
        answer = self.client.generate(
            model=self.config["ollama"]["chat_model"],
            prompt=prompt,
            system=RAG_SYSTEM_PROMPT,
            temperature=self.config["rag"].get("temperature", 0.0),
        )

        return {"query": query, "answer": answer.strip(), "sources": retrieved}
