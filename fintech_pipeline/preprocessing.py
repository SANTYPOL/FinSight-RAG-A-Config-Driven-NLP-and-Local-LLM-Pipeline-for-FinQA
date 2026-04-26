from __future__ import annotations

import re
from collections import Counter

from .constants import STOPWORDS
from .utils import normalize_text


def clean_text(text: str) -> str:
    return normalize_text(text)


def sentence_tokenize(text: str) -> list[str]:
    text = clean_text(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def word_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9₹%][A-Za-z0-9₹%&./\-]*", clean_text(text))


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token.lower() not in STOPWORDS]


def lemmatize_token(token: str) -> str:
    lower = token.lower()
    if lower.endswith("ies") and len(lower) > 4:
        return lower[:-3] + "y"
    if lower.endswith("ing") and len(lower) > 5:
        return lower[:-3]
    if lower.endswith("ed") and len(lower) > 4:
        return lower[:-2]
    if lower.endswith("es") and len(lower) > 4:
        return lower[:-2]
    if lower.endswith("s") and len(lower) > 3 and not lower.endswith("ss"):
        return lower[:-1]
    return lower


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    return [lemmatize_token(token) for token in tokens]


def prepare_analysis_text(text: str) -> dict[str, object]:
    cleaned = clean_text(text)
    sentences = sentence_tokenize(cleaned)
    tokens = word_tokenize(cleaned)
    filtered = remove_stopwords(tokens)
    lemmas = lemmatize_tokens(filtered)
    frequencies = Counter(lemmas)

    return {
        "cleaned_text": cleaned,
        "sentences": sentences,
        "tokens": tokens,
        "filtered_tokens": filtered,
        "lemmas": lemmas,
        "token_count": len(tokens),
        "sentence_count": len(sentences),
        "top_terms": frequencies.most_common(15),
    }

