# NSE Finance QA Pipeline

This repository now contains a single, config-driven pipeline for:

1. Downloading annual reports from NSE for symbols listed in a text file
2. Extracting PDF text and generating finance question-answer pairs with Ollama
3. Building a local RAG index from the generated dataset
4. Querying the dataset through the same Ollama-based pipeline

The implementation is fully local for generation and RAG. Only the NSE download step uses the network.

## Project structure

```text
.
├── config.yaml
├── api.py
├── main.py
├── requirements.txt
├── README.md
├── fintech_pipeline/
│   ├── downloader.py
│   ├── extractor.py
│   ├── ollama_client.py
│   ├── qa_generator.py
│   ├── rag.py
│   └── utils.py
├── data_downloader/
│   └── symbols/
│       └── energy.txt
├── generator/        # legacy prototype, not required for the new pipeline
└── RAG/              # legacy prototype, not required for the new pipeline
```

## Prerequisites

- Python 3.10+
- Ollama installed and running locally
- Required Ollama models pulled locally, for example:

```powershell
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

All settings are managed from [config.yaml](/D:/SEM-2/NLP/capstone/NLP_FINTECH_CAPSTONE_V01/config.yaml).

Important fields:

- `paths.symbols_file`: text file containing one NSE symbol per line
- `paths.download_dir`: where annual reports are downloaded
- `paths.cache_dir`: per-chunk raw Ollama outputs
- `paths.dataset_dir`: generated QA datasets and summaries
- `paths.rag_dir`: saved embeddings and RAG metadata
- `ollama.generation_model`: local model used to generate QA pairs
- `ollama.embedding_model`: local embedding model used for retrieval
- `ollama.chat_model`: local model used to answer RAG queries
- `nse.max_reports_per_symbol`: limit NSE downloads to the most recent N reports per symbol
- `generation.start_chunk`: start processing from this chunk index
- `generation.end_chunk`: stop processing at this chunk index

## Symbol file format

Example file: [energy.txt](/D:/SEM-2/NLP/capstone/NLP_FINTECH_CAPSTONE_V01/data_downloader/symbols/energy.txt)

```text
BPCL
RELIANCE
ONGC
```

## Usage

Run the full pipeline:

```powershell
python main.py --config config.yaml --step all
```

Run the FastAPI demo server:

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Then open:

- `http://127.0.0.1:8000/docs` for Swagger UI
- `http://127.0.0.1:8000/redoc` for ReDoc
- `http://127.0.0.1:8000/chat` for the GPT-style chat demo

Run only NSE downloads:

```powershell
python main.py --config config.yaml --step download
```

By default, only the latest 5 reports per symbol are downloaded because `nse.max_reports_per_symbol` is set to `5` in [config.yaml](/D:/SEM-2/NLP/capstone/NLP_FINTECH_CAPSTONE_V01/config.yaml). Set it to `null` to download all available reports.

Generate QA pairs from downloaded PDFs:

```powershell
python main.py --config config.yaml --step generate
```

Build the RAG index:

```powershell
python main.py --config config.yaml --step build-rag
```

Run only evaluation after the RAG index exists:

```powershell
python main.py --config config.yaml --step evaluate --eval-samples 25
```

Ask a question against the generated dataset:

```powershell
python main.py --config config.yaml --step ask --query "What was BPCL revenue and profit?"
```

Run everything and ask a question at the end:

```powershell
python main.py --config config.yaml --step all --query "What are the major capex themes for the selected companies?"
```

## FastAPI Endpoints

The project also includes a demo-ready local API in [api.py](/D:/SEM-2/NLP/capstone/NLP_FINTECH_CAPSTONE_V01/api.py).

- `GET /health`: check if the API is running
- `POST /download`: download latest NSE reports from configured symbols
- `POST /generate`: generate the finance QA dataset from downloaded PDFs
- `POST /build-rag`: build the embedding index for retrieval
- `POST /ask`: ask a finance question against the local RAG system
- `POST /evaluate`: compute exact match, token F1, and retrieval hit rate
- `POST /run-all`: run download, generation, RAG build, and one final question in one call

Example request for `/ask`:

```json
{
  "config_path": "config.yaml",
  "query": "What was BPCL PAT in 2021-22?"
}
```

This is especially useful for demonstration because Swagger UI lets you trigger the full pipeline and inspect JSON responses live in front of teachers.

## GPT-Style Demo UI

The project also includes a simple chat-style web interface served by the same FastAPI app.

- UI route: `http://127.0.0.1:8000/chat`
- It uses the existing `/ask` endpoint internally
- It shows the final answer and the retrieved supporting sources

This is useful when you want a cleaner demonstration than Swagger, while still keeping the API available for technical review.

## Outputs

After generation, the dataset folder contains:

- `<dataset_name>.json`
- `<dataset_name>.jsonl`
- `<dataset_name>.csv`
- `chunk_manifest.json`
- `chunks/<pdf_name>_chunks.json`
- `summary.json`

After RAG indexing, the RAG folder contains:

- `embeddings.npy`
- `documents.json`
- `index_metadata.json`
- `evaluation.json` after running `--step evaluate`

## Notes

- NSE sometimes rate-limits requests. If that happens, rerun the download step after a short pause.
- The downloader skips already downloaded files.
- The QA generator caches raw model outputs per chunk so reruns are much faster.
- Chunk extraction artifacts are saved to `artifacts/datasets/chunks`, so you can show the extracted chunk text directly.
- Progress bars are shown with `tqdm` during download and QA generation.
- The RAG implementation uses local dense retrieval with persisted NumPy embeddings.
- The older `generator/` and `RAG/` folders are left intact as prototypes, but the root pipeline is the supported path now.

## Evaluation metrics

The evaluation step produces a teacher-friendly report using the generated QA dataset as a benchmark:

- `exact_match`: exact normalized answer match
- `token_f1`: overlap score between predicted answer and ground-truth answer
- `retrieval_hit_rate_at_k`: whether the correct QA item was retrieved in top-k

This is not a human gold-standard benchmark, but it is useful for demonstrating retrieval quality and answer faithfulness in a project review.

## NLP Techniques Used

The project uses practical NLP methods suited for financial document understanding and question-answer generation:

- `PDF text extraction`: annual reports are converted from PDF into raw text using `pdfplumber`
- `text normalization`: whitespace cleanup and control-character removal are applied before downstream processing
- `rule-based section identification`: regex patterns are used to detect report sections such as management discussion, governance, and financial statements
- `chunking with overlap`: extracted text is broken into overlapping chunks to preserve context across boundaries
- `approximate token-length control`: chunk sizes are managed using estimated token counts for stable LLM prompting
- `LLM-based QA generation`: Ollama generates finance-focused question-answer pairs from each chunk
- `dense vector embeddings`: document records are converted into embeddings using a local Ollama embedding model
- `semantic retrieval`: user questions are matched to relevant records through vector similarity
- `retrieval-augmented generation (RAG)`: the final answer is generated from retrieved finance context only
- `automatic evaluation`: exact match, token-level F1, and retrieval hit rate are computed for demonstration and benchmarking

## Techniques Not Used

To keep the project explanation precise, the following NLP methods are not currently implemented in this pipeline:

- `Word2Vec`
- `Doc2Vec`
- `Named Entity Recognition (NER)`
- `POS tagging`
- `dependency parsing`
- `stemming or lemmatization`
- `TF-IDF or BM25 retrieval`
- `topic modeling`
- `sentiment analysis`
- `classical supervised NLP classifiers`

So the project is best described as a `document processing + chunking + embedding retrieval + local LLM QA/RAG pipeline` for finance.

## Recommended workflow

1. Put your symbols into the configured text file.
2. Make sure Ollama is running locally.
3. Run `python main.py --step all`.
4. Inspect the generated dataset in `artifacts/datasets`.
5. Use `--step ask` for interactive validation queries.
