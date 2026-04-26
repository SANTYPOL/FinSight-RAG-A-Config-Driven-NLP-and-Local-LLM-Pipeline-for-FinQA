from __future__ import annotations

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
    "from", "has", "have", "had", "he", "her", "his", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "there", "this", "to", "was",
    "were", "will", "with", "which", "who", "whom", "what", "when", "where",
    "why", "how", "than", "then", "into", "during", "over", "under", "after",
    "before", "between", "through", "per", "also", "may", "can", "such", "our",
    "your", "they", "them", "we", "you", "not", "but", "if", "about", "all",
    "any", "each", "more", "most", "other", "some", "so", "no", "nor", "too",
    "very", "s", "t", "just", "only",
}

FINANCE_TERMS = {
    "profit", "pat", "ebitda", "revenue", "sales", "turnover", "capex", "dividend",
    "share", "refinery", "marketing", "inventory", "margin", "renewable", "emission",
    "sustainability", "subsidiary", "project", "investment", "crude", "gas", "retail",
    "digital", "governance", "board", "director", "cash", "crore", "lakh", "debt",
}

ENTITY_LABELS = ("ORG", "PERSON", "MONEY", "PERCENT", "DATE", "PROJECT", "OTHER")

HYBRID_WEIGHTS = {
    "semantic": 0.7,
    "lexical": 0.3,
}

