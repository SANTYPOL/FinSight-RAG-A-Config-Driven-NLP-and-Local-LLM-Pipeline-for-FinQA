from __future__ import annotations

import argparse
import json
from pathlib import Path

from fintech_pipeline.downloader import NSEDownloader, read_symbols
from fintech_pipeline.evaluation import RAGEvaluator
from fintech_pipeline.qa_generator import QAGenerator
from fintech_pipeline.rag import RAGPipeline
from fintech_pipeline.utils import ensure_dir, read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end NSE -> Ollama QA -> RAG pipeline")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the root configuration file",
    )
    parser.add_argument(
        "--step",
        choices=["download", "analyze", "generate", "build-rag", "ask", "evaluate", "all"],
        default="all",
        help="Pipeline step to execute",
    )
    parser.add_argument(
        "--query",
        help="Question to ask after the RAG index is built",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        help="Optional limit for number of QA pairs to use during evaluation",
    )
    return parser.parse_args()


def run_download(config: dict) -> dict:
    symbols = read_symbols(config["paths"]["symbols_file"])
    ensure_dir(config["paths"]["download_dir"])
    downloader = NSEDownloader(
        output_dir=config["paths"]["download_dir"],
        sleep_seconds=config["nse"].get("sleep_seconds", 1.0),
        max_reports_per_symbol=config["nse"].get("max_reports_per_symbol"),
    )
    manifest = downloader.download_for_symbols(symbols)
    return {
        "downloaded_symbols": len(symbols),
        "reports_found": sum(len(item.get("reports", [])) for item in manifest),
    }


def run_generate(config: dict) -> dict:
    generator = QAGenerator(config)
    return generator.generate_dataset()


def run_analyze(config: dict) -> dict:
    generator = QAGenerator(config)
    return generator.analyze_documents()


def run_build_rag(config: dict) -> dict:
    rag = RAGPipeline(config)
    return rag.build_index()


def run_ask(config: dict, query: str) -> dict:
    if not query:
        raise ValueError("--query is required when --step ask is used")
    rag = RAGPipeline(config)
    return rag.ask(query)


def run_evaluate(config: dict, sample_limit: int | None) -> dict:
    evaluator = RAGEvaluator(config)
    return evaluator.evaluate(sample_limit=sample_limit)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_yaml(config_path)
    config_dir = config_path.parent

    config["paths"]["symbols_file"] = str((config_dir / config["paths"]["symbols_file"]).resolve())
    for key in ["download_dir", "cache_dir", "dataset_dir", "rag_dir"]:
        config["paths"][key] = str((config_dir / config["paths"][key]).resolve())
    analysis_path = config["paths"].get("analysis_dir", "./artifacts/analysis")
    config["paths"]["analysis_dir"] = str((config_dir / analysis_path).resolve())
    metrics_path = config["paths"].get("metrics_dir", "./artifacts/metrics")
    config["paths"]["metrics_dir"] = str((config_dir / metrics_path).resolve())

    results: dict[str, dict] = {}

    if args.step in {"download", "all"}:
        results["download"] = run_download(config)
    if args.step in {"analyze", "all"}:
        results["analyze"] = run_analyze(config)
    if args.step in {"generate", "all"}:
        results["generate"] = run_generate(config)
    if args.step in {"build-rag", "all"}:
        results["build_rag"] = run_build_rag(config)
    if args.step == "ask":
        results["ask"] = run_ask(config, args.query)
    elif args.step == "evaluate":
        results["evaluate"] = run_evaluate(config, args.eval_samples)
    elif args.step == "all" and args.query:
        results["ask"] = run_ask(config, args.query)
    elif args.step == "all":
        results["evaluate"] = run_evaluate(config, args.eval_samples)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
