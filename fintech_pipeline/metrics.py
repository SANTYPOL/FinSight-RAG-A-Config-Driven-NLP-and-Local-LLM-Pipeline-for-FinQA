from __future__ import annotations

from pathlib import Path
from typing import Any

from .plotter import (
    plot_chunk_counts,
    plot_distribution,
    plot_entity_counts,
    plot_evaluation_metrics,
    plot_keyword_summary,
    plot_processing_coverage,
)
from .utils import ensure_dir, write_json


def export_processing_metrics(metrics_dir: str | Path, report_stats: list[dict[str, Any]]) -> dict[str, Any]:
    base = ensure_dir(metrics_dir)
    plots_dir = ensure_dir(base / "plots")
    payload = {
        "report_count": len(report_stats),
        "reports": report_stats,
    }
    write_json(base / "processing_metrics.json", payload)
    plot_files = {
        "processing_coverage": plot_processing_coverage(report_stats, plots_dir),
        "chunk_counts": plot_chunk_counts(report_stats, plots_dir),
    }
    return {"metrics_file": str(base / "processing_metrics.json"), "plots": plot_files}


def export_phase1_metrics(metrics_dir: str | Path, analysis_summary: dict[str, Any]) -> dict[str, Any]:
    base = ensure_dir(metrics_dir)
    plots_dir = ensure_dir(base / "plots")
    top_entities = analysis_summary.get("top_entities", {})
    entity_counts = {label: sum(count for _, count in items) for label, items in top_entities.items()}
    write_json(base / "phase1_metrics.json", analysis_summary)
    plot_files = {
        "top_keywords": plot_keyword_summary(analysis_summary.get("top_keywords", []), plots_dir),
        "entity_counts": plot_entity_counts(entity_counts, plots_dir),
    }
    return {"metrics_file": str(base / "phase1_metrics.json"), "plots": plot_files}


def export_dataset_metrics(metrics_dir: str | Path, summary: dict[str, Any]) -> dict[str, Any]:
    base = ensure_dir(metrics_dir)
    plots_dir = ensure_dir(base / "plots")
    write_json(base / "dataset_metrics.json", summary)
    plot_files = {
        "difficulty_distribution": plot_distribution(
            summary.get("difficulty_distribution", {}),
            plots_dir,
            "difficulty_distribution.png",
            "QA Difficulty Distribution",
        ),
        "answer_type_distribution": plot_distribution(
            summary.get("answer_type_distribution", {}),
            plots_dir,
            "answer_type_distribution.png",
            "QA Answer Type Distribution",
        ),
    }
    return {"metrics_file": str(base / "dataset_metrics.json"), "plots": plot_files}


def export_evaluation_metrics(metrics_dir: str | Path, evaluation_payload: dict[str, Any]) -> dict[str, Any]:
    base = ensure_dir(metrics_dir)
    plots_dir = ensure_dir(base / "plots")
    write_json(base / "evaluation_metrics.json", evaluation_payload)
    plot_files = {
        "evaluation_metrics": plot_evaluation_metrics(evaluation_payload.get("summary", {}), plots_dir),
    }
    return {"metrics_file": str(base / "evaluation_metrics.json"), "plots": plot_files}
