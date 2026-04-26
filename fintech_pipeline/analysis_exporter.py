from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_dir, write_json


def export_chunk_analysis(base_dir: str | Path, report_stem: str, analyses: list[dict[str, Any]]) -> Path:
    target_dir = ensure_dir(Path(base_dir) / "chunks")
    path = target_dir / f"{report_stem}_analysis.json"
    write_json(path, analyses)
    return path


def export_report_analysis(base_dir: str | Path, report_stem: str, summary: dict[str, Any]) -> Path:
    target_dir = ensure_dir(Path(base_dir) / "reports")
    path = target_dir / f"{report_stem}_summary.json"
    write_json(path, summary)
    return path


def export_global_analysis(base_dir: str | Path, filename: str, payload: dict[str, Any]) -> Path:
    target_dir = ensure_dir(base_dir)
    path = Path(target_dir) / filename
    write_json(path, payload)
    return path

