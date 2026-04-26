from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pdfplumber

from .utils import normalize_whitespace


@dataclass
class Chunk:
    chunk_id: str
    symbol: str
    source_file: str
    page_range: str
    section: str
    text: str
    token_estimate: int

    def to_dict(self) -> dict:
        return asdict(self)


SECTION_PATTERNS = [
    r"^(?:MANAGEMENT DISCUSSION|BOARD'S REPORT|DIRECTORS' REPORT|FINANCIAL HIGHLIGHTS)",
    r"^(?:CORPORATE GOVERNANCE|AUDITOR\S* REPORT|STANDALONE FINANCIAL STATEMENTS)",
    r"^[A-Z][A-Z0-9 &,\-()/.]{8,}$",
]


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def is_section_header(line: str) -> bool:
    stripped = normalize_whitespace(line)
    if not stripped:
        return False
    return any(re.match(pattern, stripped) for pattern in SECTION_PATTERNS)


def extract_chunks_from_pdf(
    pdf_path: str | Path,
    symbol: str,
    chunk_tokens: int,
    overlap_tokens: int,
    min_chunk_tokens: int,
) -> list[Chunk]:
    chunks, _ = extract_chunks_with_stats_from_pdf(
        pdf_path=pdf_path,
        symbol=symbol,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        min_chunk_tokens=min_chunk_tokens,
    )
    return chunks


def extract_chunks_with_stats_from_pdf(
    pdf_path: str | Path,
    symbol: str,
    chunk_tokens: int,
    overlap_tokens: int,
    min_chunk_tokens: int,
) -> tuple[list[Chunk], dict[str, Any]]:
    path = Path(pdf_path)
    paragraphs: list[tuple[int, str, str]] = []
    current_section = "General"
    total_pages = 0

    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        for page_index, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            for line in page_text.splitlines():
                clean_line = normalize_whitespace(line)
                if not clean_line:
                    continue
                if is_section_header(clean_line):
                    current_section = clean_line[:100]
                    continue
                if len(clean_line) >= 40:
                    paragraphs.append((page_index, current_section, clean_line))

    if not paragraphs:
        return [], {
            "source_file": str(path),
            "symbol": symbol,
            "total_pdf_pages": total_pages,
            "total_extracted_paragraphs": 0,
            "total_chunks_created_before_capping": 0,
            "chunks_kept_after_capping": 0,
            "final_covered_page_range": None,
        }

    chunks: list[Chunk] = []
    buffer_pages: list[int] = []
    buffer_text: list[str] = []
    buffer_tokens = 0
    buffer_section = paragraphs[0][1]

    def flush() -> None:
        nonlocal buffer_pages, buffer_text, buffer_tokens, buffer_section
        if not buffer_text:
            return
        joined = " ".join(buffer_text).strip()
        token_estimate = estimate_tokens(joined)
        if token_estimate < min_chunk_tokens:
            return
        page_range = str(buffer_pages[0]) if len(set(buffer_pages)) == 1 else f"{buffer_pages[0]}-{buffer_pages[-1]}"
        chunks.append(
            Chunk(
                chunk_id=f"{symbol.lower()}_{path.stem}_chunk_{len(chunks):04d}",
                symbol=symbol,
                source_file=str(path),
                page_range=page_range,
                section=buffer_section,
                text=joined,
                token_estimate=token_estimate,
            )
        )

    for page_num, section, paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)
        should_flush = False

        if buffer_text and section != buffer_section:
            should_flush = True
        elif buffer_text and buffer_tokens + paragraph_tokens > chunk_tokens:
            should_flush = True

        if should_flush:
            prior_paragraphs = list(buffer_text)
            flush()

            overlap: list[str] = []
            running_tokens = 0
            for item in reversed(prior_paragraphs):
                item_tokens = estimate_tokens(item)
                if running_tokens + item_tokens > overlap_tokens:
                    break
                overlap.insert(0, item)
                running_tokens += item_tokens

            buffer_text = overlap
            buffer_tokens = sum(estimate_tokens(item) for item in buffer_text)
            buffer_pages = [page_num]
            buffer_section = section

        if not buffer_text:
            buffer_section = section

        buffer_text.append(paragraph)
        buffer_tokens += paragraph_tokens
        buffer_pages.append(page_num)

    flush()
    covered_ranges = [chunk.page_range for chunk in chunks]
    final_range = covered_ranges[-1] if covered_ranges else None
    stats = {
        "source_file": str(path),
        "symbol": symbol,
        "total_pdf_pages": total_pages,
        "total_extracted_paragraphs": len(paragraphs),
        "total_chunks_created_before_capping": len(chunks),
        "chunks_kept_after_capping": len(chunks),
        "final_covered_page_range": final_range,
    }
    return chunks, stats
