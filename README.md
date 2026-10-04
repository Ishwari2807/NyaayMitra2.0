# NyaayMitra — Combined Tanvi + Ishwari Architecture

This is the merged project with **Tanvi's project as the base**, selected stronger pieces from the other implementation, and the V2 architecture applied to the weak layers.

## Final architecture

`Ingestion → Structure/Chunking → Embedding + ChromaDB → BM25 → RRF → Reranker → Situation/Issue Analysis → Evidence Gate → Grounded Generation → Claim-Level Grounding → Response`

## What was changed

- Common normalized document representation preserves pages, raw blocks, tables and metadata.
- Structure-first legal chunking; structural split before semantic fallback.
- Parent-section and provenance metadata added to chunks.
- Footnotes/amendment text is preserved instead of silently discarded.
- Vector/BM25 retrieval uses `is_retrievable` rather than requiring `legal_section`, so verified non-statutory material can remain searchable.
- Situation analysis now produces potential issues and retrieval information needs.
- Evidence Gate checks relevance, sufficiency, issue coverage and provenance rather than using definition markers/20% word overlap as its foundation.
- Grounded generation retains evidence-only generation and deterministic source attachment.
- Grounding Validator works at claim/list-item level rather than only field-level word overlap.
- LangGraph is the single production orchestration path; `AdvisorPipeline` is only a compatibility wrapper.
- Your structured user-facing response fields are added to the final answer contract.
- Embedding and reranker models are intentionally not changed without benchmarking.

## 1. Open in VS Code

Open this folder as the project root.

## 2. Create the environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, run the test with `.venv\Scripts\python.exe final_test.py` after installing dependencies.

## 3. Run the final test

```powershell
python final_test.py
```

or double-click `run_final_test.bat`.

The final test is deliberately offline: it does **not** need Ollama, an API key, a downloaded embedding model, or a Chroma index. It verifies the integrated architecture using deterministic test doubles.

## 4. Run the real knowledge-base pipeline

Put your legal source files under:

`data/raw/`

Supported parsers currently include PDF, DOCX, CSV, XLSX and TXT.

Run ingestion:

```powershell
python -m backend.ingestion.run_ingestion
```

Build the vector index:

```powershell
python -m backend.rag.build_index
```

The BM25 retriever reads the generated `data/chunks/*.json` files.

## 5. Run the live end-to-end test

Install and start Ollama, then pull the model configured in `.env`/environment (default `llama3.2:1b`). Then:

```powershell
python live_test.py
```

## Embedding model policy

The current embedding model remains `sentence-transformers/all-MiniLM-L6-v2`. Do **not** replace it just because a legal-domain model is larger. Benchmark candidates on the NyaayMitra evaluation dataset using Recall@5, Recall@10 and MRR before rebuilding the vector index.

## Important

No real API keys or secrets are included in this ZIP. Copy `.env.example` to `.env` only if you need local configuration. The `.env` file is ignored by Git.
