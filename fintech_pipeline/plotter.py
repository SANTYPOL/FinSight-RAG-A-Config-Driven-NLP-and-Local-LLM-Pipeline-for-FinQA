from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .utils import ensure_dir


def _save_plot(target: Path) -> str:
    plt.tight_layout()
    plt.savefig(target, dpi=180, bbox_inches="tight")
    plt.close()
    return str(target)


def plot_processing_coverage(report_stats: list[dict[str, Any]], output_dir: str | Path) -> str | None:
    if not report_stats:
        return None

    ensure_dir(output_dir)
    labels = [Path(item["source_file"]).stem[:18] for item in report_stats]
    total_pages = [item.get("total_pdf_pages", 0) for item in report_stats]
    covered_pages = []
    for item in report_stats:
        page_range = item.get("final_covered_page_range")
        if not page_range:
            covered_pages.append(0)
        elif "-" in page_range:
            covered_pages.append(int(str(page_range).split("-")[-1]))
        else:
            covered_pages.append(int(page_range))

    plt.figure(figsize=(12, 5))
    x = range(len(labels))
    plt.bar(x, total_pages, label="Total PDF Pages", color="#cfe8f3")
    plt.bar(x, covered_pages, label="Final Covered Page", color="#2a9d8f")
    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("Pages")
    plt.title("PDF Coverage by Processed Report")
    plt.legend()
    return _save_plot(Path(output_dir) / "processing_coverage.png")


def plot_chunk_counts(report_stats: list[dict[str, Any]], output_dir: str | Path) -> str | None:
    if not report_stats:
        return None

    ensure_dir(output_dir)
    labels = [Path(item["source_file"]).stem[:18] for item in report_stats]
    before = [item.get("total_chunks_created_before_capping", 0) for item in report_stats]
    after = [item.get("chunks_kept_after_capping", 0) for item in report_stats]

    plt.figure(figsize=(12, 5))
    x = range(len(labels))
    width = 0.36
    plt.bar([i - width / 2 for i in x], before, width=width, label="Before Capping", color="#f4a261")
    plt.bar([i + width / 2 for i in x], after, width=width, label="After Capping", color="#264653")
    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("Chunks")
    plt.title("Chunks Created vs Chunks Kept")
    plt.legend()
    return _save_plot(Path(output_dir) / "chunk_counts.png")


def plot_distribution(distribution: dict[str, int], output_dir: str | Path, filename: str, title: str) -> str | None:
    if not distribution:
        return None
    ensure_dir(output_dir)
    labels = list(distribution.keys())
    values = list(distribution.values())
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color="#457b9d")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    return _save_plot(Path(output_dir) / filename)


def plot_keyword_summary(top_keywords: list[tuple[str, int]] | list[list[Any]], output_dir: str | Path) -> str | None:
    if not top_keywords:
        return None
    ensure_dir(output_dir)
    labels = [item[0] for item in top_keywords[:10]]
    values = [item[1] for item in top_keywords[:10]]
    plt.figure(figsize=(10, 5))
    plt.barh(labels[::-1], values[::-1], color="#8ecae6")
    plt.xlabel("Frequency")
    plt.title("Top Finance Keywords")
    return _save_plot(Path(output_dir) / "top_keywords.png")


def plot_entity_counts(entity_counts: dict[str, int], output_dir: str | Path) -> str | None:
    if not entity_counts:
        return None
    ensure_dir(output_dir)
    labels = list(entity_counts.keys())
    values = list(entity_counts.values())
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color="#e76f51")
    plt.ylabel("Extracted Entities")
    plt.title("Entity Counts by Type")
    plt.xticks(rotation=20, ha="right")
    return _save_plot(Path(output_dir) / "entity_counts.png")


def plot_evaluation_metrics(summary: dict[str, Any], output_dir: str | Path) -> str | None:
    if not summary:
        return None
    ensure_dir(output_dir)
    labels = ["Exact Match", "Token F1", "Hit Rate@k"]
    values = [
        float(summary.get("exact_match", 0.0)),
        float(summary.get("token_f1", 0.0)),
        float(summary.get("retrieval_hit_rate_at_k", 0.0)),
    ]
    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, color=["#2a9d8f", "#e9c46a", "#264653"])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("RAG Evaluation Metrics")
    return _save_plot(Path(output_dir) / "evaluation_metrics.png")

