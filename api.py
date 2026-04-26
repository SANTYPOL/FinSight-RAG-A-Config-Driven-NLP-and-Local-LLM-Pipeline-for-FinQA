from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from fintech_pipeline.downloader import NSEDownloader, read_symbols
from fintech_pipeline.evaluation import RAGEvaluator
from fintech_pipeline.qa_generator import QAGenerator
from fintech_pipeline.rag import RAGPipeline
from fintech_pipeline.utils import read_yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    resolved = Path(config_path).resolve()
    config = read_yaml(resolved)
    config_dir = resolved.parent
    config["paths"]["symbols_file"] = str((config_dir / config["paths"]["symbols_file"]).resolve())
    for key in ["download_dir", "cache_dir", "dataset_dir", "rag_dir"]:
        config["paths"][key] = str((config_dir / config["paths"][key]).resolve())
    analysis_path = config["paths"].get("analysis_dir", "./artifacts/analysis")
    config["paths"]["analysis_dir"] = str((config_dir / analysis_path).resolve())
    metrics_path = config["paths"].get("metrics_dir", "./artifacts/metrics")
    config["paths"]["metrics_dir"] = str((config_dir / metrics_path).resolve())
    return config


app = FastAPI(
    title="NSE Finance QA Pipeline API",
    description="Local API for NSE download, QA generation, RAG indexing, query answering, and evaluation.",
    version="1.0.0",
)
UI_DIR = Path(__file__).resolve().parent / "ui"


class ConfigurableRequest(BaseModel):
    config_path: str = Field(default="config.yaml", description="Path to pipeline config")


class AskRequest(ConfigurableRequest):
    query: str = Field(..., min_length=3, description="User question to ask the RAG pipeline")


class EvaluateRequest(ConfigurableRequest):
    eval_samples: int | None = Field(default=None, ge=1, description="Optional number of samples to evaluate")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/chat")


@app.get("/api-info")
def api_info() -> dict[str, Any]:
    return {
        "message": "NSE Finance QA Pipeline API is running",
        "docs": "/docs",
        "chat": "/chat",
        "endpoints": [
            "/health",
            "/download",
            "/analyze",
            "/generate",
            "/build-rag",
            "/ask",
            "/evaluate",
            "/run-all",
        ],
    }


@app.get("/chat")
def chat_ui() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/ui/styles.css")
def chat_styles() -> FileResponse:
    return FileResponse(UI_DIR / "styles.css", media_type="text/css")


@app.get("/ui/app.js")
def chat_script() -> FileResponse:
    return FileResponse(UI_DIR / "app.js", media_type="application/javascript")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/download")
def download_reports(request: ConfigurableRequest) -> dict[str, Any]:
    try:
        config = load_config(request.config_path)
        symbols = read_symbols(config["paths"]["symbols_file"])
        downloader = NSEDownloader(
            output_dir=config["paths"]["download_dir"],
            sleep_seconds=config["nse"].get("sleep_seconds", 1.0),
            max_reports_per_symbol=config["nse"].get("max_reports_per_symbol"),
        )
        manifest = downloader.download_for_symbols(symbols)
        return {
            "downloaded_symbols": len(symbols),
            "reports_found": sum(len(item.get("reports", [])) for item in manifest),
            "manifest": manifest,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate")
def generate_dataset(request: ConfigurableRequest) -> dict[str, Any]:
    try:
        config = load_config(request.config_path)
        generator = QAGenerator(config)
        return generator.generate_dataset()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/analyze")
def analyze_documents(request: ConfigurableRequest) -> dict[str, Any]:
    try:
        config = load_config(request.config_path)
        generator = QAGenerator(config)
        return generator.analyze_documents()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/build-rag")
def build_rag_index(request: ConfigurableRequest) -> dict[str, Any]:
    try:
        config = load_config(request.config_path)
        rag = RAGPipeline(config)
        return rag.build_index()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
def ask_question(request: AskRequest) -> dict[str, Any]:
    try:
        config = load_config(request.config_path)
        rag = RAGPipeline(config)
        return rag.ask(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/evaluate")
def evaluate_rag(request: EvaluateRequest) -> dict[str, Any]:
    try:
        config = load_config(request.config_path)
        evaluator = RAGEvaluator(config)
        return evaluator.evaluate(sample_limit=request.eval_samples)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/run-all")
def run_all(request: AskRequest) -> dict[str, Any]:
    try:
        config = load_config(request.config_path)
        symbols = read_symbols(config["paths"]["symbols_file"])
        downloader = NSEDownloader(
            output_dir=config["paths"]["download_dir"],
            sleep_seconds=config["nse"].get("sleep_seconds", 1.0),
            max_reports_per_symbol=config["nse"].get("max_reports_per_symbol"),
        )
        download_manifest = downloader.download_for_symbols(symbols)

        generator = QAGenerator(config)
        generate_result = generator.generate_dataset()

        rag = RAGPipeline(config)
        build_result = rag.build_index()
        ask_result = rag.ask(request.query)

        return {
            "download": {
                "downloaded_symbols": len(symbols),
                "reports_found": sum(len(item.get("reports", [])) for item in download_manifest),
            },
            "generate": generate_result,
            "build_rag": build_result,
            "ask": ask_result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
