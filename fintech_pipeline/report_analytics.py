from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def build_global_analysis(report_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    keyword_counter: Counter[str] = Counter()
    entity_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for summary in report_summaries:
        for keyword, count in summary.get("top_keywords", []):
            keyword_counter[keyword] += count
        for label, items in summary.get("top_entities", {}).items():
            for entity, count in items:
                entity_counter[label][entity] += count

    return {
        "report_count": len(report_summaries),
        "top_keywords": keyword_counter.most_common(25),
        "top_entities": {label: counter.most_common(20) for label, counter in entity_counter.items()},
    }

