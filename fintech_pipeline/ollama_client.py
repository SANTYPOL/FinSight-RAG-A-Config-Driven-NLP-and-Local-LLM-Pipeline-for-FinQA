from __future__ import annotations

from typing import Any

import requests


class OllamaAPIError(RuntimeError):
    """Raised when Ollama requests fail in a user-actionable way."""


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, model: str, prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return (data.get("response") or "").strip()

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        errors: list[str] = []

        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": texts},
                timeout=self.timeout,
            )
            if response.ok:
                data = response.json()
                embeddings = data.get("embeddings", [])
                if embeddings:
                    return embeddings
                errors.append("/api/embed returned no embeddings")
            else:
                errors.append(self._format_http_error("/api/embed", response, model))
        except requests.RequestException as exc:
            errors.append(f"/api/embed request failed: {exc}")

        legacy_vectors: list[list[float]] = []
        legacy_failed = False
        for text in texts:
            try:
                legacy_response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=self.timeout,
                )
                if not legacy_response.ok:
                    legacy_failed = True
                    errors.append(self._format_http_error("/api/embeddings", legacy_response, model))
                    break

                legacy_data = legacy_response.json()
                embedding = legacy_data.get("embedding")
                if not embedding:
                    legacy_failed = True
                    errors.append("/api/embeddings returned no embedding vector")
                    break
                legacy_vectors.append(embedding)
            except requests.RequestException as exc:
                legacy_failed = True
                errors.append(f"/api/embeddings request failed: {exc}")
                break

        if legacy_vectors and not legacy_failed:
            return legacy_vectors

        error_message = [
            f"Could not create embeddings with Ollama model '{model}'.",
            "Possible reasons:",
            "1. The embedding model is not pulled locally.",
            "2. Your Ollama version supports only one of the embedding endpoints.",
            "3. The configured model name is not an embedding-capable model.",
            "4. Ollama is running but returned an API/version-specific error.",
            "",
            "Recommended checks:",
            f"- Run: ollama pull {model}",
            "- Run: ollama list",
            "- Confirm Ollama is running at the configured base URL.",
            "",
            "API details:",
            *[f"- {item}" for item in errors],
        ]
        raise OllamaAPIError("\n".join(error_message))

    def _format_http_error(self, endpoint: str, response: requests.Response, model: str) -> str:
        try:
            payload = response.json()
            detail = payload.get("error") or payload
        except ValueError:
            detail = response.text.strip()

        return (
            f"{endpoint} returned HTTP {response.status_code} for model '{model}'"
            + (f": {detail}" if detail else "")
        )
