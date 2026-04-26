from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .analysis_exporter import export_chunk_analysis, export_global_analysis, export_report_analysis
from .extractor import Chunk, extract_chunks_with_stats_from_pdf
from .metrics import export_dataset_metrics, export_phase1_metrics, export_processing_metrics
from .nlp_analysis import analyze_chunks, summarize_analyses
from .ollama_client import OllamaClient
from .report_analytics import build_global_analysis
from .utils import ensure_dir, normalize_text, read_json, safe_json_array, write_json

SYSTEM_PROMPT = """You are a financial dataset builder.
Create high-quality finance question-answer pairs from annual report excerpts.
Return only a valid JSON array with no markdown and no explanation.

Each object must contain:
- category
- difficulty
- question
- answer
- context_hint
- answer_type

Rules:
- Ground every answer strictly in the supplied text.
- Questions must be self-contained and end with a question mark.
- Answers must be concise and include exact figures when present.
- Prefer professional finance questions over generic metadata questions.
- Avoid trivial questions unless the chunk is primarily corporate-identification content.
- Never answer with placeholders such as "Not mentioned in the excerpt".
- difficulty must be one of: Easy, Medium, Hard.
- answer_type must be one of: Factual, Quantitative, Comparative, Analytical.
- context_hint should be a short useful phrase from the section, not empty.
- category must be one of: Corporate Overview, Financial Performance, Profitability, Capital Expenditure, Strategy & Growth, ESG & Sustainability, Operations, Marketing & Sales, Governance, Board & Management, Subsidiaries & JVs, Digital Transformation, Risk & Compliance, Dividend & Shareholding, Projects & Investments, Other.
- Generate between 2 and 5 pairs per chunk depending on the information density.
"""

USER_TEMPLATE = """Symbol: {symbol}
Section: {section}
Pages: {pages}
Source file: {source_file}
Keywords: {keywords}
Entities: {entities}

Excerpt:
{text}
"""

VALID_DIFFICULTY = {"Easy", "Medium", "Hard"}
VALID_ANSWER_TYPES = {"Factual", "Quantitative", "Comparative", "Analytical"}
VALID_CATEGORIES = {
    "Corporate Overview",
    "Financial Performance",
    "Profitability",
    "Capital Expenditure",
    "Strategy & Growth",
    "ESG & Sustainability",
    "Operations",
    "Marketing & Sales",
    "Governance",
    "Board & Management",
    "Subsidiaries & JVs",
    "Digital Transformation",
    "Risk & Compliance",
    "Dividend & Shareholding",
    "Projects & Investments",
    "Other",
}

CATEGORY_ALIASES = {
    "company info": "Corporate Overview",
    "financials": "Financial Performance",
    "financial performance": "Financial Performance",
    "financial performance ": "Financial Performance",
    "company performance": "Financial Performance",
    "profitability": "Profitability",
    "business strategy": "Strategy & Growth",
    "corporate strategy": "Strategy & Growth",
    "company strategy": "Strategy & Growth",
    "renewable energy": "ESG & Sustainability",
    "corporate social responsibility": "ESG & Sustainability",
    "management": "Board & Management",
    "board and management": "Board & Management",
    "digital transformation": "Digital Transformation",
}

WEAK_ANSWERS = {
    "not mentioned in the excerpt",
    "not mentioned",
    "not available",
    "not available in the excerpt",
    "not provided",
    "unknown",
}


def log_pdf_processing_stats(pdf_name: str, stats: dict[str, Any]) -> None:
    print(
        "\n".join(
            [
                f"[stats] {pdf_name}",
                f"  pages                  : {stats.get('total_pdf_pages', 0)}",
                f"  extracted_paragraphs   : {stats.get('total_extracted_paragraphs', 0)}",
                f"  chunks_before_capping  : {stats.get('total_chunks_created_before_capping', 0)}",
                f"  chunks_after_capping   : {stats.get('chunks_kept_after_capping', 0)}",
                f"  final_covered_range    : {stats.get('final_covered_page_range') or 'NA'}",
            ]
        )
    )


def print_processing_summary(report_stats: list[dict[str, Any]], title: str) -> None:
    if not report_stats:
        return

    print(f"\n[summary] {title}")
    print("report | pages | paragraphs | chunks_before | chunks_after | covered_range")
    print("-" * 92)
    for item in report_stats:
        report_name = Path(item.get("source_file", "")).stem[:24]
        pages = item.get("total_pdf_pages", 0)
        paragraphs = item.get("total_extracted_paragraphs", 0)
        before = item.get("total_chunks_created_before_capping", 0)
        after = item.get("chunks_kept_after_capping", 0)
        covered = item.get("final_covered_page_range") or "NA"
        print(
            f"{report_name:<24} | {pages:>5} | {paragraphs:>10} | {before:>13} | {after:>12} | {covered}"
        )


def canonicalize_category(category: str, question: str, answer: str, chunk: Chunk) -> str:
    raw = normalize_text(category).strip(" .,:;").lower()
    if raw in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[raw]
    if category in VALID_CATEGORIES:
        return category

    text = " ".join([question, answer, chunk.section]).lower()
    if any(term in text for term in ["director", "chairman", "managing director", "independent director", "board"]):
        return "Board & Management"
    if any(term in text for term in ["profit", "pat", "ebitda", "margin", "revenue", "sales", "turnover"]):
        return "Financial Performance"
    if any(term in text for term in ["capex", "investment", "project", "expansion"]):
        return "Projects & Investments"
    if any(term in text for term in ["digital", "technology", "anubhav"]):
        return "Digital Transformation"
    if any(term in text for term in ["net zero", "renewable", "sustainability", "emission", "csr"]):
        return "ESG & Sustainability"
    if any(term in text for term in ["subsidiary", "joint venture", "jv", "merged"]):
        return "Subsidiaries & JVs"
    return "Other"


def infer_answer_type(question: str, answer: str, answer_type: str) -> str:
    cleaned = normalize_text(answer_type).title()
    if cleaned in VALID_ANSWER_TYPES:
        if cleaned == "Quantitative" and not any(char.isdigit() for char in answer):
            cleaned = "Factual"
        return cleaned

    question_lower = question.lower()
    if any(term in question_lower for term in ["how much", "how many", "what was", "what were", "how far", "how long"]):
        if any(char.isdigit() for char in answer):
            return "Quantitative"
    if any(term in question_lower for term in ["compare", "difference", "versus", "evolving into", "which of the following"]):
        return "Comparative"
    if any(term in question_lower for term in ["why", "reason", "impact", "expected outcome", "significance", "how did"]):
        return "Analytical"
    return "Factual"


def clean_context_hint(context_hint: str, chunk: Chunk) -> str:
    hint = normalize_text(context_hint)
    if not hint or hint.lower() in {"page", "pages", "section", "general"}:
        return normalize_text(chunk.section)
    if len(hint.split()) > 18:
        return normalize_text(chunk.section)
    return hint


def looks_too_generic(question: str, chunk: Chunk) -> bool:
    q = question.lower()
    generic_patterns = [
        "where has the annual report",
        "what is the cin",
        "what is the bse scrip code",
        "what is the website",
    ]
    if any(pattern in q for pattern in generic_patterns):
        section = chunk.section.lower()
        return "board" not in section and "management" not in section and "corporate" not in section
    return False


def normalize_pair(pair: dict[str, Any], chunk: Chunk, analysis: dict[str, Any] | None = None) -> dict[str, Any] | None:
    question = normalize_text(pair.get("question", ""))
    answer = normalize_text(pair.get("answer", ""))
    difficulty = normalize_text(pair.get("difficulty", "Medium")).title()

    if len(question.split()) < 5 or len(answer.split()) < 1 or not question.endswith("?"):
        return None
    if answer.lower() in WEAK_ANSWERS:
        return None
    if looks_too_generic(question, chunk):
        return None

    if difficulty not in VALID_DIFFICULTY:
        difficulty = "Medium"

    cleaned = {
        "id": None,
        "symbol": chunk.symbol,
        "category": canonicalize_category(
            str(pair.get("category", "Other")),
            question,
            answer,
            chunk,
        ),
        "difficulty": difficulty,
        "question": question,
        "answer": answer,
        "context_hint": clean_context_hint(str(pair.get("context_hint", "")), chunk),
        "answer_type": infer_answer_type(question, answer, str(pair.get("answer_type", "Factual"))),
        "source_section": normalize_text(chunk.section),
        "source_pages": normalize_text(chunk.page_range),
        "source_chunk_id": chunk.chunk_id,
        "source_file": chunk.source_file,
        "source_keywords": (analysis or {}).get("keywords", []),
        "source_entities": (analysis or {}).get("entities", {}),
        "token_count": (analysis or {}).get("token_count", chunk.token_estimate),
    }
    return cleaned


def validate_pairs(pairs: list[dict[str, Any]], chunk: Chunk, analysis: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for pair in pairs:
        cleaned = normalize_pair(pair, chunk, analysis)
        if cleaned:
            validated.append(cleaned)
    return validated


def deduplicate_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for pair in pairs:
        key = normalize_text(f"{pair['symbol']}::{pair['question']}".lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)

    for index, pair in enumerate(unique, start=1):
        pair["id"] = index
    return unique


def export_dataset(pairs: list[dict[str, Any]], output_dir: str | Path, stem: str) -> dict[str, str]:
    directory = ensure_dir(output_dir)
    json_path = directory / f"{stem}.json"
    jsonl_path = directory / f"{stem}.jsonl"
    csv_path = directory / f"{stem}.csv"

    write_json(json_path, pairs)

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(f"{json.dumps(pair, ensure_ascii=False)}\n")

    fields = [
        "id",
        "symbol",
        "category",
        "difficulty",
        "question",
        "answer",
        "context_hint",
        "answer_type",
        "source_section",
        "source_pages",
        "source_chunk_id",
        "source_file",
        "source_keywords",
        "source_entities",
        "token_count",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)

    return {"json": str(json_path), "jsonl": str(jsonl_path), "csv": str(csv_path)}


class QAGenerator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.client = OllamaClient(
            base_url=config["ollama"]["base_url"],
            timeout=config["ollama"].get("timeout", 120),
        )

    def analyze_documents(self) -> dict[str, Any]:
        extraction_config = self.config["generation"]["extraction"]
        generation_config = self.config["generation"]
        input_dir = Path(self.config["paths"]["download_dir"])
        dataset_dir = ensure_dir(self.config["paths"]["dataset_dir"])
        analysis_dir = ensure_dir(self.config["paths"].get("analysis_dir", dataset_dir.parent / "analysis"))
        metrics_dir = ensure_dir(self.config["paths"].get("metrics_dir", dataset_dir.parent / "metrics"))
        chunks_dir = ensure_dir(dataset_dir / "chunks")

        pdf_paths = sorted(input_dir.rglob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files were found in download directory: {input_dir}")

        report_summaries: list[dict[str, Any]] = []
        total_chunks = 0
        processing_stats: list[dict[str, Any]] = []

        for pdf_path in tqdm(pdf_paths, desc="Analyzing PDFs", unit="pdf"):
            symbol = pdf_path.parent.name.upper()
            chunks, stats = extract_chunks_with_stats_from_pdf(
                pdf_path=pdf_path,
                symbol=symbol,
                chunk_tokens=extraction_config["chunk_tokens"],
                overlap_tokens=extraction_config["overlap_tokens"],
                min_chunk_tokens=extraction_config["min_chunk_tokens"],
            )
            before_capping = len(chunks)

            start_chunk = generation_config.get("start_chunk", 0)
            end_chunk = generation_config.get("end_chunk")
            if start_chunk:
                chunks = chunks[start_chunk:]
            if end_chunk is not None:
                relative_end = max(0, end_chunk - start_chunk + 1)
                chunks = chunks[:relative_end]
            if generation_config.get("max_chunks_per_pdf"):
                chunks = chunks[: generation_config["max_chunks_per_pdf"]]

            stats["total_chunks_created_before_capping"] = before_capping
            stats["chunks_kept_after_capping"] = len(chunks)
            stats["final_covered_page_range"] = chunks[-1].page_range if chunks else None
            processing_stats.append(stats)
            log_pdf_processing_stats(pdf_path.name, stats)

            analyses = analyze_chunks(chunks)
            analysis_by_chunk = {item["chunk_id"]: item for item in analyses}
            enriched_manifest = []
            for chunk in chunks:
                chunk_data = chunk.to_dict()
                analysis = analysis_by_chunk.get(chunk.chunk_id, {})
                chunk_data["keywords"] = analysis.get("keywords", [])
                chunk_data["entities"] = analysis.get("entities", {})
                chunk_data["sentence_count"] = analysis.get("sentence_count", 0)
                chunk_data["token_count"] = analysis.get("token_count", chunk.token_estimate)
                enriched_manifest.append(chunk_data)

            write_json(chunks_dir / f"{pdf_path.stem}_chunks.json", enriched_manifest)
            summary = summarize_analyses(analyses, pdf_path.name)
            export_chunk_analysis(analysis_dir, pdf_path.stem, analyses)
            export_report_analysis(analysis_dir, pdf_path.stem, summary)
            report_summaries.append(summary)
            total_chunks += len(chunks)

        global_summary = build_global_analysis(report_summaries)
        export_global_analysis(analysis_dir, "summary.json", global_summary)
        export_global_analysis(
            analysis_dir,
            "report_summaries.json",
            {"reports": report_summaries},
        )
        processing_export = export_processing_metrics(metrics_dir, processing_stats)
        phase1_export = export_phase1_metrics(metrics_dir, global_summary)
        print_processing_summary(processing_stats, "Analysis Coverage Overview")
        return {
            "reports_analyzed": len(report_summaries),
            "total_chunks": total_chunks,
            "analysis_dir": str(analysis_dir),
            "metrics_dir": str(metrics_dir),
            "processing_metrics": processing_export,
            "phase1_metrics": phase1_export,
            "top_keywords": global_summary.get("top_keywords", [])[:10],
        }

    def generate_dataset(self) -> dict[str, Any]:
        extraction_config = self.config["generation"]["extraction"]
        generation_config = self.config["generation"]
        input_dir = Path(self.config["paths"]["download_dir"])
        cache_dir = ensure_dir(self.config["paths"]["cache_dir"])
        dataset_dir = ensure_dir(self.config["paths"]["dataset_dir"])
        analysis_dir = ensure_dir(self.config["paths"].get("analysis_dir", dataset_dir.parent / "analysis"))
        metrics_dir = ensure_dir(self.config["paths"].get("metrics_dir", dataset_dir.parent / "metrics"))
        chunks_dir = ensure_dir(dataset_dir / "chunks")

        pdf_paths = sorted(input_dir.rglob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(
                f"No PDF files were found in download directory: {input_dir}"
            )

        all_pairs: list[dict[str, Any]] = []
        chunk_manifest: list[dict[str, Any]] = []
        report_summaries: list[dict[str, Any]] = []
        processing_stats: list[dict[str, Any]] = []

        pdf_iterator = tqdm(pdf_paths, desc="Generating QA from PDFs", unit="pdf")
        for pdf_path in pdf_iterator:
            symbol = pdf_path.parent.name.upper()
            chunks, stats = extract_chunks_with_stats_from_pdf(
                pdf_path=pdf_path,
                symbol=symbol,
                chunk_tokens=extraction_config["chunk_tokens"],
                overlap_tokens=extraction_config["overlap_tokens"],
                min_chunk_tokens=extraction_config["min_chunk_tokens"],
            )
            before_capping = len(chunks)

            start_chunk = generation_config.get("start_chunk", 0)
            end_chunk = generation_config.get("end_chunk")
            if start_chunk:
                chunks = chunks[start_chunk:]
            if end_chunk is not None:
                relative_end = max(0, end_chunk - start_chunk + 1)
                chunks = chunks[:relative_end]
            if generation_config.get("max_chunks_per_pdf"):
                chunks = chunks[: generation_config["max_chunks_per_pdf"]]

            stats["total_chunks_created_before_capping"] = before_capping
            stats["chunks_kept_after_capping"] = len(chunks)
            stats["final_covered_page_range"] = chunks[-1].page_range if chunks else None
            processing_stats.append(stats)
            log_pdf_processing_stats(pdf_path.name, stats)

            chunk_analyses = analyze_chunks(chunks)
            analysis_by_chunk = {item["chunk_id"]: item for item in chunk_analyses}
            report_summary = summarize_analyses(chunk_analyses, pdf_path.name)
            report_summaries.append(report_summary)

            enriched_manifest = []
            for chunk in chunks:
                chunk_data = chunk.to_dict()
                analysis = analysis_by_chunk.get(chunk.chunk_id, {})
                chunk_data["keywords"] = analysis.get("keywords", [])
                chunk_data["entities"] = analysis.get("entities", {})
                chunk_data["sentence_count"] = analysis.get("sentence_count", 0)
                chunk_data["token_count"] = analysis.get("token_count", chunk.token_estimate)
                enriched_manifest.append(chunk_data)

            chunk_manifest.extend(enriched_manifest)
            write_json(chunks_dir / f"{pdf_path.stem}_chunks.json", [chunk.to_dict() for chunk in chunks])
            export_chunk_analysis(analysis_dir, pdf_path.stem, chunk_analyses)
            export_report_analysis(analysis_dir, pdf_path.stem, report_summary)

            chunk_iterator = tqdm(chunks, desc=f"{pdf_path.stem} chunks", unit="chunk", leave=False)
            for chunk in chunk_iterator:
                analysis = analysis_by_chunk.get(chunk.chunk_id, {})
                cache_path = cache_dir / f"{chunk.chunk_id}.json"
                if cache_path.exists():
                    raw_pairs = read_json(cache_path)
                else:
                    prompt = USER_TEMPLATE.format(
                        symbol=chunk.symbol,
                        section=chunk.section,
                        pages=chunk.page_range,
                        source_file=chunk.source_file,
                        keywords=", ".join(analysis.get("keywords", [])[:8]) or "None",
                        entities=json.dumps(analysis.get("entities", {}), ensure_ascii=False),
                        text=chunk.text,
                    )
                    raw_output = self.client.generate(
                        model=self.config["ollama"]["generation_model"],
                        prompt=prompt,
                        system=SYSTEM_PROMPT,
                        temperature=generation_config.get("temperature", 0.2),
                    )
                    raw_pairs = safe_json_array(raw_output)
                    write_json(cache_path, raw_pairs)

                all_pairs.extend(validate_pairs(raw_pairs, chunk, analysis))

        final_pairs = deduplicate_pairs(all_pairs)
        write_json(dataset_dir / "chunk_manifest.json", chunk_manifest)
        global_summary = build_global_analysis(report_summaries)
        export_global_analysis(analysis_dir, "summary.json", global_summary)
        export_global_analysis(
            analysis_dir,
            "report_summaries.json",
            {"reports": report_summaries},
        )
        processing_export = export_processing_metrics(metrics_dir, processing_stats)
        phase1_export = export_phase1_metrics(metrics_dir, global_summary)
        dataset_paths = export_dataset(final_pairs, dataset_dir, self.config["dataset"]["name"])

        summary = {
            "dataset_name": self.config["dataset"]["name"],
            "total_pairs": len(final_pairs),
            "symbols": sorted({pair["symbol"] for pair in final_pairs}),
            "difficulty_distribution": dict(Counter(pair["difficulty"] for pair in final_pairs)),
            "answer_type_distribution": dict(Counter(pair["answer_type"] for pair in final_pairs)),
            "files": dataset_paths,
            "analysis_dir": str(analysis_dir),
            "metrics_dir": str(metrics_dir),
        }
        write_json(dataset_dir / "summary.json", summary)
        dataset_export = export_dataset_metrics(metrics_dir, summary)
        summary["processing_metrics"] = processing_export
        summary["phase1_metrics"] = phase1_export
        summary["dataset_metrics"] = dataset_export
        print_processing_summary(processing_stats, "Generation Coverage Overview")
        write_json(dataset_dir / "summary.json", summary)
        return summary
