# Vector Index Benchmarking Framework

**Performance Evaluation of Vector Indexing Techniques for AI Search Systems**

COMP.8157 Advanced Database Topics, 2026S · University of Windsor, School of Computer Science
Instructor: Dr. Andreas S. Maniatis · **Group 4**

---

## Project links

| | |
|---|---|
| **Source code repository** | [github.com/Devz0201/COMP8157-Vector-Indexing](https://github.com/Devz0201/COMP8157-Vector-Indexing) |
| **Project management tool** | [Hive board](https://app.hive.com/workspace/JbYCPPC6L8DCGyzKG?actionViewId=W4yJq7vXwthAk2DqF&tabId=aprxRosGk2vxPJixC&projectId=48q28Z9gpJfRKWDJs) — see [`docs/PROJECT_MANAGEMENT.md`](docs/PROJECT_MANAGEMENT.md) |
| **Run it without installing anything** | [Open in Google Colab](https://colab.research.google.com/github/Devz0201/COMP8157-Vector-Indexing/blob/main/notebooks/VecBench_Colab.ipynb) · or [open a GitHub Codespace](https://codespaces.new/Devz0201/COMP8157-Vector-Indexing) |
| **Dataset** | A Million News Headlines — [doi:10.7910/DVN/SYBGZL](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SYBGZL) (CC0), provenance in [`data/DATASET.md`](data/DATASET.md) |
| **Database schema** | [`db/schema.sql`](db/schema.sql) — connection instructions [below](#the-results-database) |

### Documentation (deliverable D.4)

| Document | File |
|---|---|
| D.4.1 User Requirements and Analysis | [`docs/D.4.1_User_Requirements_and_Analysis_Group4.docx`](docs/) |
| D.4.2 Design Document | [`docs/D.4.2_Design_Document_Group4.docx`](docs/) |
| D.4.4 Deployment Document | [`docs/D.4.4_Deployment_Document_Group4.docx`](docs/) · Markdown mirror: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |
| D.4.5 User Guide | [`docs/D.4.5_User_Guide_Group4.docx`](docs/) · Markdown mirror: [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) |
| D.5 Final Project Report | [`docs/D.5_Final_Project_Report_Group4.docx`](docs/) · PDF: [`docs/D.5_Final_Project_Report_Group4.pdf`](docs/) |
| D.3.1 Midterm Status Report | [`docs/D3_1_Midterm_Project_Status_Report_IEEE.pdf`](docs/) |
| Individual contribution reports | [`docs/contributions/`](docs/contributions/) |
| Design figures (architecture, data flow, ERD) | [`docs/figures/`](docs/figures/) |

---

## What this is

A workload-consistent benchmarking framework that compares the three dominant
vector-index families — **Flat (exact)**, **IVF (Inverted File)**, and
**HNSW (graph-based)** — under *identical* conditions, and measures the metrics
that actually matter for an AI search backend:

| Metric | What it tells you |
|---|---|
| **Query latency** (mean / p50 / p95) | How fast a single search returns |
| **Recall@K** | How accurate the approximate results are vs exact search |
| **Index memory** | RAM/storage footprint of the index |
| **Build time** | Cost of constructing the index |
| **Scalability** | How all of the above change as the dataset grows |

FAISS implements all three families, but does not prescribe a fair way to
compare them — and most published benchmarks change the corpus, the embedding
model, or the hardware between methods. This framework fixes every variable
except the index structure, so a measured difference is attributable to the
index and nothing else.

```
Data Ingestion → Embedding → Indexing → Query Engine → Evaluation & Visualization → Results DB
 (Component 1)   (Comp. 2)   (Comp. 3)    (Comp. 4)         (Component 5)          (Component 6)
```

---

## Quick start

The fastest path with nothing installed is the **[Colab notebook](https://colab.research.google.com/github/Devz0201/COMP8157-Vector-Indexing/blob/main/notebooks/VecBench_Colab.ipynb)** —
*Runtime → Run all* reproduces every result in about ten minutes.

To run locally (Python 3.10+, CPU only, no GPU anywhere in the pipeline):

```bash
git clone https://github.com/Devz0201/COMP8157-Vector-Indexing.git
cd COMP8157-Vector-Indexing

pip install -r requirements.txt          # core: all prebuilt wheels, no compilation
pip install sentence-transformers        # optional: the real SBERT path
```

Then either run everything at once:

```bash
./run_all.sh              # Linux / macOS   (run_all.sh --offline to skip the SBERT steps)
run_all.bat               # Windows
```

or run the experiments individually:

```bash
# 1. Scalability sweep -- synthetic vectors, no downloads (1k -> 100k)
python run_benchmark.py --config configs/scalability.yaml

# 2. Real Sentence-BERT benchmark on 20,000 news headlines
#    (data/headlines.txt ships with the repo, so this works straight away)
python run_benchmark.py --config configs/text_sbert.yaml

# 3. Parameter-tuning sweep: how IVF nprobe / HNSW efSearch trade speed for accuracy
python tune_parameters.py --config configs/tuning.yaml

# 4. Browse everything that has been run
python db_cli.py summary
```

Every run writes a CSV table and PNG charts under `results/`, **and** records
itself in the SQLite results database under `db/`.

---

## Dataset

The real-data run uses **"A Million News Headlines"** (ABC News, 1.24M
headlines, 2003–2021), released under CC0. A deterministic 20,000-headline
sample is committed as `data/headlines.txt`, so nothing needs downloading to
reproduce the reported results.

To rebuild it from source — or to benchmark a different sample size — one
command fetches the raw CSV from Harvard Dataverse (no account, no API key) and
prepares it:

```bash
python fetch_dataset.py --n 20000        # exactly reproduces the committed file
python fetch_dataset.py --n 50000        # a larger corpus
```

Full provenance, licence, and preparation details: [`data/DATASET.md`](data/DATASET.md).

---

## The results database

Every run is appended to a SQLite database at **`db/vecbench_results.db`**,
alongside the CSVs. The CSVs are the per-run artefact; the database is what
makes runs *comparable* — "how does HNSW at 100k compare with last week's run
on the same machine" is a `WHERE` clause rather than a manual diff of a
directory full of files.

### Connecting to it

SQLite is embedded, so there is **no server, no host, no port, and no
credentials** — the database is the file. Three ways in:

```bash
# 1. The bundled CLI (no extra install)
python db_cli.py runs                    # every run, newest first
python db_cli.py show latest             # one run in full, with its config
python db_cli.py results --run latest --method HNSW
python db_cli.py logs --run latest --level ERROR

# 2. The sqlite3 shell, if it is installed
sqlite3 db/vecbench_results.db "SELECT * FROM v_method_summary;"

# 3. From Python -- nothing to install, sqlite3 is in the standard library
python -c "import sqlite3, pandas as pd; \
print(pd.read_sql('SELECT * FROM v_scalability', sqlite3.connect('db/vecbench_results.db')))"
```

The database creates itself from `db/schema.sql` on first use, so a fresh clone
needs no setup step.

### Schema at a glance

| Table | One row per | Key columns |
|---|---|---|
| `run` | benchmark invocation | config (verbatim), dataset, embedding backend, host, Python/FAISS versions, status |
| `measurement` | (run, method, dataset size) | latency mean/p50/p95, recall@K, memory, build time, QPS |
| `tuning_point` | parameter setting swept | method, knob (`nprobe`/`efSearch`), value, recall, latency |
| `run_log` | log line emitted during a run | timestamp, level, pipeline stage, message |

Four views (`v_results`, `v_method_summary`, `v_scalability`, `v_tuning_curve`)
pre-join the run context onto the measurements so common questions do not
require writing the join by hand. Constraints are doing real work: `recall_at_k`
is `CHECK`ed into `[0,1]`, `method` into the three implemented families, and
`ON DELETE CASCADE` means dropping a bad run takes its measurements and logs
with it.

### Filtering and searching

The interesting queries combine filters. "Which configurations answered in
under a millisecond *while still* holding recall at or above 0.95?" is:

```bash
python db_cli.py results --min-recall 0.95 --max-latency 1.0 --sort latency
```

and anything the fixed commands do not cover is available as read-only SQL:

```bash
python db_cli.py sql "SELECT method, AVG(recall_at_k) FROM measurement GROUP BY method"
```

---

## Configuration

Everything is driven by a YAML config — no code changes between experiments:

```yaml
mode: single            # 'single' or 'scalability'
k: 10                   # top-K to retrieve / evaluate
per_query: true         # per-query timing (avoids batch/cache bias)

data:
  source: text          # 'text' (corpus) or 'random' (synthetic)
  path: data/headlines.txt
  target_size: 20000    # number of documents to benchmark
  n_query: 200          # held-out queries
  seed: 42

embedding:
  backend: sbert        # 'sbert' | 'hashing' | 'synthetic'
  model_name: all-MiniLM-L6-v2

methods: [flat, ivf, hnsw]

params:
  ivf:  { nlist: 100, nprobe: 32 }                     # IVF speed/recall knobs
  hnsw: { M: 32, ef_construction: 80, ef_search: 64 }  # HNSW knobs
```

| Config | What it runs |
|---|---|
| `configs/scalability.yaml` | dataset-size sweep 1k → 100k, offline synthetic vectors |
| `configs/text_sbert.yaml` | real Sentence-BERT over the news-headlines corpus |
| `configs/tuning.yaml` | IVF `nprobe` and HNSW `efSearch` sweeps |
| `configs/ci_smoke.yaml` | the two-size version CI runs on every push |

### Why three embedding backends?

The proposal specifies **Sentence-BERT**, and that is the production path
(`embedding.backend: sbert`). But SBERT needs a one-time model download, so the
framework also ships offline backends — the *entire pipeline runs anywhere*,
including air-gapped machines and CI:

| Backend | Needs download? | Use for |
|---|---|---|
| `sbert` | yes | **Real semantic retrieval** (the proposal's method) |
| `hashing` | no | Offline runs on real text (lower quality, fully reproducible) |
| `synthetic` | no | Large-scale scalability sweeps with clustered vectors |

The indexing, query, evaluation, and visualization layers are **identical**
regardless of how the vectors were produced — only the embedder is swapped.

---

## Results

**Real Sentence-BERT run** — 20,000 ABC news headlines, 200 held-out queries, `k=10`:

| Method | Configuration | Latency mean / p95 (ms) | QPS | Recall@10 | Memory (MB) | Build (s) |
|---|---|---|---|---|---|---|
| Flat | `Flat` | 1.818 / 2.252 | 550 | 1.0000 | 29.3 | 0.011 |
| IVF | `IVF(nlist=100,nprobe=32)` | 1.168 / 2.110 | 856 | 0.9625 | 29.6 | 0.223 |
| HNSW | `HNSW(M=32,efS=64)` | 0.451 / 0.850 | 2218 | 0.9950 | 34.5 | 2.729 |

HNSW answers **4.0× faster than exact search** while keeping recall at 0.995; IVF is 1.6× faster at 0.963. Flat's recall is 1.000 by definition — it is the ground truth the other two are scored against, not a competitor.

**Scalability sweep** — synthetic clustered vectors, 1k → 100k, `k=10`:

| Metric | 1,000 | 5,000 | 20,000 | 50,000 | 100,000 |
|---|---|---|---|---|---|
| **Flat** latency (ms) | 0.050 | 0.402 | 1.249 | 4.402 | 8.027 |
| Flat recall@10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **IVF** latency (ms) | 0.032 | 0.052 | 0.244 | 0.521 | 1.017 |
| IVF recall@10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **HNSW** latency (ms) | 0.057 | 0.055 | 0.153 | 0.264 | 0.338 |
| HNSW recall@10 | 1.000 | 0.995 | 0.911 | 0.803 | 0.696 |

Flat latency grows linearly with N, as an exhaustive scan must. IVF holds recall at 1.000 across the whole range, but that is a property of the input rather than of IVF: the synthetic corpus has 64 well-separated Gaussian clusters, so the 8 cells `nprobe` visits almost always contain the true neighbours. On the real SBERT embeddings above, the same defaults only reach 0.839. HNSW is the fastest at every scale, but at a **fixed** `ef_search = 64` its recall decays from 1.000 at 1k to 0.696 at 100k — the clearest result in the project: an approximate index's parameters are not a one-time choice, they have to be raised as the data grows.

**Parameter tuning** (`tune_parameters.py`) quantifies the speed/accuracy knob:
past a certain point, searching harder only adds latency without adding recall.

**IVF** — 20,000 SBERT-embedded headlines

| `nprobe` | Recall@10 | Latency (ms) |
|---|---|---|
| 1 | 0.5055 | 0.040 |
| 2 | 0.6275 | 0.067 |
| 4 | 0.7450 | 0.126 |
| 8 | 0.8390 | 0.244 |
| 16 | 0.9100 | 0.491 |
| 32 | 0.9625 | 0.726 |
| 64 | 0.9940 | 1.724 |
| 100 | 1.0000 | 2.540 |

**HNSW** — same corpus

| `efSearch` | Recall@10 | Latency (ms) |
|---|---|---|
| 8 | 0.9770 | 0.092 |
| 16 | 0.9865 | 0.142 |
| 32 | 0.9905 | 0.255 |
| 64 | 0.9970 | 0.345 |
| 128 | 0.9985 | 0.426 |
| 256 | 0.9990 | 0.766 |

The two curves behave differently, and that difference is the practical finding. IVF has to search 32 of its 100 cells before it clears 0.95 on real data, and buys the last 3.7 points of recall at 3.5× the latency. HNSW is already at 0.977 with `efSearch = 8`; everything past 64 costs latency for less than half a point of recall. This is what fixed `configs/text_sbert.yaml` after the deployment incident recorded in D.4.4.

**Key insight:** there is no universal winner. On real data, IVF and HNSW both
match Flat's accuracy while being far faster — but the right *settings* must be
tuned, and (as the scalability sweep shows) raised as the dataset grows.

Charts: `results/scalability/scalability.png`, `results/sbert/comparison_bars.png`,
`results/tuning/tuning_tradeoff.png`.

---

## Project structure

```
COMP8157-Vector-Indexing/
├── run_benchmark.py          # CLI entry point: single + scalability runs
├── tune_parameters.py        # CLI entry point: IVF/HNSW parameter sweeps
├── db_cli.py                 # CLI entry point: browse the results database
├── fetch_dataset.py          # downloads + prepares the headlines corpus
├── prepare_headlines.py      # converts the raw CSV into the input format
├── run_all.sh / run_all.bat  # one-command reproduction of every result
├── pyproject.toml            # project metadata, dependency set, tool config
├── requirements.txt
├── configs/                  # YAML experiment configs
├── data/
│   ├── DATASET.md            # provenance, licence, preparation steps
│   ├── headlines.txt         # 20,000-headline benchmark corpus (committed)
│   └── sample_corpus.txt     # example input format
├── db/
│   ├── schema.sql            # the results database schema
│   └── vecbench_results.db   # generated on first run
├── docs/                     # D.4.1, D.4.2, D.4.4, D.4.5, midterm report, PM
├── notebooks/                # Colab notebook (zero-install run)
├── results/                  # generated tables + charts
├── tests/                    # pytest suite (43 tests)
└── src/
    ├── data_ingestion.py     # Component 1: load, clean, segment
    ├── embeddings.py         # Component 2: SBERT / hashing / synthetic
    ├── indexing.py           # Component 3: Flat / IVF / HNSW (FAISS)
    ├── query_engine.py       # Component 4: query exec + Component 5 metrics
    ├── benchmark.py          # orchestrator (single + scalability)
    ├── visualization.py      # comparative + scalability charts
    └── results_db.py         # Component 6: persistence to SQLite
```

---

## Tests

```bash
pip install pytest ruff
pytest                 # 43 tests: pipeline, indexing, evaluation, database
ruff check .           # lint
```

CI runs the suite plus a real end-to-end benchmark on Ubuntu and Windows across
Python 3.10, 3.11, and 3.13 on every push — the project's reproducibility claim
is only worth something if it holds on a machine nobody on the team owns.

---

## Extending the framework

- **New index** → subclass `BaseIndex` in `indexing.py`, register it in
  `build_index()` (e.g. IVF-PQ, ScaNN, LSH), and add it to the `method` CHECK
  constraint in `db/schema.sql`.
- **New embedder** → add a `BaseEmbedder` subclass in `embeddings.py`.
- **New metric** → extend `MethodMetrics` and `evaluate()` in `query_engine.py`,
  and add the column to the `measurement` table.
- **New dataset** → point `data.path` at a text file, one document per line.

---

## Reproducibility

All randomness is seeded (`data.seed`), embeddings are deterministic, and a
warm-up phase plus per-query timing removes execution-order and caching bias,
so repeated runs of the same config yield consistent measurements. The results
database stores each run's full config alongside the host, Python version, and
FAISS version, so two rows can be checked for comparability rather than assumed
to be comparable.

One caveat, measured rather than assumed (D.5 §VIII-E). The corpus sample, the
held-out queries, the embeddings, the serialized index size, and Flat and IVF
recall all reproduce exactly. **HNSW recall does not.** FAISS inserts points
into the HNSW graph in parallel, so the neighbour lists depend on thread
interleaving and the graph differs slightly between builds. Three builds at
n=20,000, same seed, gave recall 0.889 / 0.894 / 0.884; the same config on the
machine that produced the tables above gave 0.911. Pinning construction to one
thread removes the spread entirely:

```bash
OMP_NUM_THREADS=1 python run_benchmark.py --config configs/scalability.yaml
# three runs -> 0.8985 / 0.8985 / 0.8985
```

Do this before comparing two nearby `ef_search` settings, where the differences
are smaller than the spread.

---

## Team — Group 4

| Name | Role | GitHub | Email |
|---|---|---|---|
| Devu Babu Sheeja | Data & Embeddings Lead | [`Devz0201`](https://github.com/Devz0201) | babushe@uwindsor.ca |
| Ashwin Senthur Pandian | Vector Indexing & Tuning Lead | [`ashwin-0707`](https://github.com/ashwin-0707) | senthur@uwindsor.ca |
| Rishi G. Patel | Query & Evaluation Lead | [`Rishihi29`](https://github.com/Rishihi29) | patel3zc@uwindsor.ca |
| Nikhil Goud Nathi | Experiments & Visualization Lead | [`nikhilnathi2003`](https://github.com/nikhilnathi2003) | nathin@uwindsor.ca |

Licensed under the [MIT License](LICENSE).
