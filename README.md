# Vector Index Benchmarking Framework

**Performance Evaluation of Vector Indexing Techniques for AI Search Systems**
COMP.8157 — Advanced Database Topics, University of Windsor, 2026S — Group 4

> **Status note (as of this commit):** this repository reflects work completed
> through the D.3.1 Midterm checkpoint. The data pipeline, embeddings,
> benchmarking automation, visualization, and two of the three indexes (Flat
> and IVF) are implemented and tested. The **HNSW index** and the
> **parameter-tuning experiments** are still in progress locally and will be
> pushed in a follow-up commit once finished and tested. See
> `D3_1_Midterm_Project_Status_Report.pdf` for the full task breakdown.

A benchmarking framework that compares vector index strategies — currently
**Flat Search** (exact baseline) and **IVF** (Inverted File Index) — under
identical conditions using FAISS and Sentence-BERT embeddings, measuring query
latency, recall@K, memory usage, index build time, and scalability.

---

## Quick start

```bash
pip install -r requirements.txt
pip install sentence-transformers   # for the real SBERT run

# scalability sweep (synthetic vectors, no downloads)
python run_benchmark.py --config configs/scalability.yaml

# real-data run (needs a prepared corpus, see below)
python prepare_headlines.py --csv abcnews-date-text.csv --out data/headlines.txt --n 20000
python run_benchmark.py --config configs/text_sbert.yaml
```

Outputs (CSV tables + PNG charts) are written to `results/`.

## Dataset

Real-data runs use **"A Million News Headlines"** (ABC News, ~1.1M headlines,
CC0 licence): https://www.kaggle.com/datasets/therohk/million-headlines

## Project structure

```
├── run_benchmark.py        # CLI entry point
├── prepare_headlines.py    # dataset preparation script
├── requirements.txt
├── configs/                # experiment configs (flat + ivf only, this snapshot)
├── data/                   # sample corpus
└── src/
    ├── data_ingestion.py   # load, clean, segment text
    ├── embeddings.py       # Sentence-BERT / offline embedding backends
    ├── indexing.py         # Flat, IVF (HNSW to follow)
    ├── query_engine.py     # query execution + timing + recall@K
    ├── benchmark.py        # orchestration (single + scalability modes)
    └── visualization.py    # comparative + scalability charts
```

## Team & contributions

| Member | Area |
|---|---|
| Devu Babu Sheeja | Data & Embeddings (`data_ingestion.py`, `prepare_headlines.py`, `embeddings.py`) |
| Ashwin Senthur Pandian | Vector Indexing & Tuning (`indexing.py`; HNSW + `tune_parameters.py` in progress) |
| Rishi Gaurangkumar Patel | Query & Evaluation (`query_engine.py`, `benchmark.py`) |
| Nikhil Goud Nathi | Experiments & Visualization (`run_benchmark.py`, `visualization.py`, configs) |

See `D3_1_Midterm_Project_Status_Report.pdf` for the full task breakdown and
`D3_1_1_Team_Contribution_Form_Group4.docx` for individual contributions.
