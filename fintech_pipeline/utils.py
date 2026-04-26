from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_json(path: str | Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    return cleaned.strip("_").lower() or "item"


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def repair_common_mojibake(text: str) -> str:
    replacements = {
        "â‚¹": "₹",
        "â€œ": '"',
        "â€": '"',
        "â€™": "'",
        "â€“": "-",
        "â€”": "-",
        "Â ": " ",
        "Â": "",
        "Ã—": "x",
    }
    fixed = text
    for bad, good in replacements.items():
        fixed = fixed.replace(bad, good)
    return fixed


def normalize_text(text: str) -> str:
    return normalize_whitespace(repair_common_mojibake(str(text)))


def safe_json_array(raw_text: str) -> list[dict[str, Any]]:
    raw_text = raw_text.strip()
    raw_text = re.sub(r"```json\s*|```", "", raw_text, flags=re.IGNORECASE)

    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", raw_text)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
