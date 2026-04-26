from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from .constants import ENTITY_LABELS, FINANCE_TERMS
from .extractor import Chunk
from .preprocessing import prepare_analysis_text
from .utils import normalize_text


MONEY_PATTERN = re.compile(r"(₹\s?[\d,]+(?:\.\d+)?\s*(?:crore|lakh|million|billion)?|\b[\d,]+(?:\.\d+)?\s*crore\b)", re.IGNORECASE)
PERCENT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%")
DATE_PATTERN = re.compile(r"\b(?:20\d{2}-\d{2}|20\d{2}|FY\s?\d{2,4}[-/]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b", re.IGNORECASE)
ORG_PATTERN = re.compile(r"\b(?:BPCL|ONGC|RELIANCE|BORL|BPRL|NSE|Government of India|Govt\. of Kerala|Bina Refinery|Kochi Refinery)\b")
PERSON_PATTERN = re.compile(r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)?\s?[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")
PROJECT_PATTERN = re.compile(r"\b(?:Project\s+[A-Z][A-Za-z0-9]+|Net Zero|City Gas Distribution|Ethylene Cracker|Polypropylene unit)\b", re.IGNORECASE)


def _dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = normalize_text(value)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def extract_entities(text: str) -> dict[str, list[str]]:
    entities = {
        "ORG": _dedupe_preserve(ORG_PATTERN.findall(text)),
        "PERSON": _dedupe_preserve(PERSON_PATTERN.findall(text)),
        "MONEY": _dedupe_preserve(MONEY_PATTERN.findall(text)),
        "PERCENT": _dedupe_preserve(PERCENT_PATTERN.findall(text)),
        "DATE": _dedupe_preserve(DATE_PATTERN.findall(text)),
        "PROJECT": _dedupe_preserve(PROJECT_PATTERN.findall(text)),
    }
    entities["OTHER"] = []
    return {label: entities.get(label, []) for label in ENTITY_LABELS}


def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    prepared = prepare_analysis_text(text)
    lemmas = prepared["lemmas"]
    counts = Counter(lemmas)

    boosted: list[tuple[str, float]] = []
    for token, count in counts.items():
        score = float(count)
        if token in FINANCE_TERMS:
            score += 2.5
        if any(char.isdigit() for char in token):
            score -= 0.5
        if len(token) < 3:
            score -= 1
        boosted.append((token, score))

    boosted.sort(key=lambda item: (-item[1], item[0]))
    return [token for token, _ in boosted[:top_n]]


def extract_keyphrases(text: str, top_n: int = 5) -> list[str]:
    prepared = prepare_analysis_text(text)
    bigrams = zip(prepared["lemmas"], prepared["lemmas"][1:])
    counter = Counter(" ".join(pair) for pair in bigrams if all(len(word) > 2 for word in pair))
    ranked = [phrase for phrase, _ in counter.most_common(top_n)]
    return ranked


def analyze_chunk(chunk: Chunk) -> dict[str, Any]:
    prepared = prepare_analysis_text(chunk.text)
    entities = extract_entities(chunk.text)
    keywords = extract_keywords(chunk.text, top_n=10)
    keyphrases = extract_keyphrases(chunk.text, top_n=5)

    return {
        "chunk_id": chunk.chunk_id,
        "symbol": chunk.symbol,
        "source_file": chunk.source_file,
        "page_range": chunk.page_range,
        "section": normalize_text(chunk.section),
        "token_estimate": chunk.token_estimate,
        "token_count": prepared["token_count"],
        "sentence_count": prepared["sentence_count"],
        "keywords": keywords,
        "keyphrases": keyphrases,
        "entities": entities,
        "top_terms": prepared["top_terms"],
        "cleaned_text": prepared["cleaned_text"],
    }


def analyze_chunks(chunks: list[Chunk]) -> list[dict[str, Any]]:
    return [analyze_chunk(chunk) for chunk in chunks]


def summarize_analyses(chunk_analyses: list[dict[str, Any]], report_name: str) -> dict[str, Any]:
    keyword_counter: Counter[str] = Counter()
    entity_counter: dict[str, Counter[str]] = defaultdict(Counter)
    total_tokens = 0
    total_sentences = 0

    for analysis in chunk_analyses:
        keyword_counter.update(analysis.get("keywords", []))
        total_tokens += int(analysis.get("token_count", 0))
        total_sentences += int(analysis.get("sentence_count", 0))
        for label, values in analysis.get("entities", {}).items():
            entity_counter[label].update(values)

    return {
        "report_name": report_name,
        "chunk_count": len(chunk_analyses),
        "total_tokens": total_tokens,
        "total_sentences": total_sentences,
        "top_keywords": keyword_counter.most_common(15),
        "top_entities": {label: counter.most_common(10) for label, counter in entity_counter.items()},
    }

