# D.4.5 User Guide

*Performance Evaluation of Vector Indexing Techniques for AI Search Systems*

> This is the Markdown mirror of **`D.4.5_User_Guide_Group4.docx`**, kept in the repository so the
> document is readable on GitHub without downloading it. The Word file is the submitted
> deliverable; if the two ever disagree, the Word file is authoritative.

COMP.8157 Advanced Database Topics, 2026S · University of Windsor, School of Computer Science
Instructor: Dr. Andreas S. Maniatis · **Group 4**

| Name | Role | Email |
|---|---|---|
| Devu Babu Sheeja | Data & Embeddings Lead | babushe@uwindsor.ca |
| Ashwin Senthur Pandian | Vector Indexing & Tuning Lead | senthur@uwindsor.ca |
| Rishi G. Patel | Query & Evaluation Lead | patel3zc@uwindsor.ca |
| Nikhil Goud Nathi | Experiments & Visualization Lead | nathin@uwindsor.ca |

---

## 1. Overview

### 1.1 Product purpose

The Vector Index Benchmarking Framework answers one question: for a given corpus and a given accuracy requirement, which vector index should you actually run? Anyone building an AI search backend has to choose between exact search and the approximate families that trade a little accuracy for a lot of speed, and the published comparisons are rarely decisive because they change the corpus, the embedding model, or the hardware between the methods they compare. This tool fixes every one of those variables and varies only the index, so the numbers it reports are attributable to the index and nothing else.

You give it a corpus and a configuration file. It embeds the text once, builds a Flat (exact) index, an IVF index, and an HNSW index over those same vectors, runs the same held-out queries against all three under the same conditions, and reports four things per method: query latency as mean, median, and 95th percentile; recall@K scored against the exact Flat result; the index's memory footprint; and how long it took to build. It can repeat that across a range of dataset sizes to show how the picture changes as data grows, and it can sweep each approximate index's speed/accuracy knob to show what tuning buys.

What it is not: it is not a search service. There is no always-on API, no web interface, and no query endpoint to point an application at. It is a measurement tool that you run when you need a decision, and it exits when it is finished. Output is written to disk as CSV tables and PNG charts, and appended to a small SQLite database so runs can be compared with each other later.

### 1.2 System requirements

|  | Minimum | Notes |
|---|---|---|
| Operating system | Windows 10+, macOS 12+, or current Linux | Developed on Windows 11, verified in CI on Ubuntu and Windows |
| Python | 3.10 or newer | Verified on 3.10, 3.11, and 3.13 |
| CPU | 2 cores, x86-64 or ARM64 | No GPU is used anywhere in the pipeline |
| RAM | 4 GB | The 100,000-vector sweep peaks a little above 1 GB |
| Free disk | 2 GB | Dependencies, the model cache, and generated results |
| Network | Only for the one-time model and dataset downloads | Everything else runs offline |

If you would rather not install anything at all, skip straight to section 2.1: the project ships a Google Colab notebook that runs the whole framework in the browser.

## 2. Getting Started

### 2.1 Installation

Option A — run it in the cloud, nothing installed. Open notebooks/VecBench_Colab.ipynb from the repository in Google Colab and choose Runtime → Run all. The notebook clones the project, installs its dependencies, runs all three experiments, displays the charts, and queries the results database. It takes about ten minutes end to end. A GitHub Codespace works the same way and is configured by .devcontainer/devcontainer.json.

Option B — install locally. Three commands:

```bash
git clone https://github.com/Devz0201/COMP8157-Vector-Indexing.git
cd COMP8157-Vector-Indexing
pip install -r requirements.txt
```

That installs the core dependencies — FAISS, NumPy, pandas, matplotlib, scikit-learn, and PyYAML — all as prebuilt wheels, so nothing is compiled on your machine. If you also want the real Sentence-BERT embeddings described in the proposal, add one more:

```bash
pip install sentence-transformers
```

This one is kept separate on purpose. It pulls in PyTorch and downloads a language model the first time it runs, and the framework is deliberately usable without it — the offline embedding backends drive exactly the same indexing, query, and evaluation code, so you can exercise the full pipeline on a machine with no network access.

### 2.2 Initial setup

There is no login. The framework has no user accounts, no authentication, and no credentials of any kind, because it is a local command-line tool with nothing listening on a port and no shared state between users. If you can run Python on the machine, you can run the benchmark; there is nothing to sign in to.

Nor is there a database server to configure. The results database is SQLite, which is embedded rather than hosted — the database is a single file at db/vecbench_results.db, with no host, no port, no username, and no password. It creates itself from db/schema.sql the first time anything writes to it. You can confirm the whole environment is ready with one command, which prints the tables and views it has created:

```bash
python db_cli.py init
```

The benchmark corpus is already in place: data/headlines.txt holds a 20,000-headline sample of the ABC News dataset, committed to the repository so the real-data benchmark runs immediately after cloning. You only need to fetch anything if you want a different sample size, in which case one command downloads the source and rebuilds it:

```bash
python fetch_dataset.py --n 50000
```

### 2.3 Your first run

Start with the scalability sweep. It uses synthetic vectors, so it needs no downloads and no optional dependencies, and it finishes in a couple of minutes:

```bash
python run_benchmark.py --config configs/scalability.yaml
```

You will see each index reporting its latency, recall, memory, and build time at each dataset size as it goes, then a summary table. When it finishes it tells you where it wrote the CSV and the charts, and which run number it was recorded as. To run everything the report is based on — the sweep, the real Sentence-BERT benchmark, and the parameter tuning — use the bundled launcher instead:

```bash
run_all.bat # Windows
./run_all.sh # Linux / macOS (add --offline to skip the SBERT steps)
```

## 3. Core Features

### 3.1 Running an experiment

Everything about an experiment lives in a YAML configuration file, never in the source code. Four are provided, and you point the runner at whichever you want:

| Configuration | What it does | Runtime |
|---|---|---|
| configs/scalability.yaml | Sweeps dataset size from 1,000 to 100,000 synthetic vectors across all three index families | ~2 minutes |
| configs/text_sbert.yaml | The real benchmark: 20,000 news headlines embedded with Sentence-BERT | ~3 minutes |
| configs/tuning.yaml | Sweeps IVF's nprobe and HNSW's efSearch to map the speed/accuracy trade-off | ~4 minutes |
| configs/ci_smoke.yaml | A two-size version of the sweep, used to check an installation quickly | ~10 seconds |

To change an experiment, edit the YAML — the dataset, the embedding backend, the value of k, which methods to compare, and every index parameter are all fields in that file. Changing HNSW's graph degree or IVF's cluster count never requires touching the source code, which means the exact configuration behind any result you report can be committed alongside it.

### 3.2 Reading the results

Each run writes a CSV table and a set of PNG charts under results/, in the directory named by the config's outdir field. The CSV has one row per index method per dataset size, with the metric columns; the charts are the same data drawn as comparison bars, a latency-versus-recall scatter, and — for a sweep — one line chart per metric against dataset size.

Read them together with one rule in mind: Flat's recall is always exactly 1.000, because Flat is the exact search that every other method is scored against. It is the reference, not a competitor. What you are looking for is how much latency IVF and HNSW save, and how much recall that costs.

### 3.3 Viewing logs

Every run records its own execution log into the same database as its results, tagged with a severity level and the pipeline stage that emitted it. That means a run's history does not scroll away with the terminal, and when something looks wrong the log and the numbers it produced are in one place. To read the log of the most recent run:

```bash
python db_cli.py logs --run latest
```

The log can be narrowed in four ways, and the filters combine. Use --level to show only one severity, --stage to follow a single phase of the pipeline, --grep to search the message text, and --tail to see only the last few lines:

```bash
python db_cli.py logs --run latest --level WARN # only the warnings
python db_cli.py logs --run latest --stage index # only index construction
python db_cli.py logs --grep "HNSW" --tail 20 # last 20 lines mentioning HNSW
python db_cli.py logs --run 3 --level ERROR # did run 3 fail, and where?
```

Severity levels mean what you would expect. INFO records normal progress — how many documents were ingested, which index was built and how long it took, what each method measured. WARN marks something that is not a failure but is worth your attention, most often an approximate index whose recall has fallen below 0.95, which is the single most common reason a result looks wrong. ERROR is written when a run raises an exception, along with the exception type and message, immediately before the run is marked failed. A quick health check across every run ever executed on the machine is therefore one command, and it should return nothing:

```bash
python db_cli.py logs --level ERROR
```

### 3.4 Filtering and searching data

The CSV files answer “what did this run report”. The database answers the questions that span runs, which is where the interesting comparisons live — how today's HNSW numbers compare with last week's on the same machine, or which of everything you have ever run actually met your requirements. Start by listing the runs:

```bash
python db_cli.py runs # every run, newest first
python db_cli.py runs --experiment tuning # only parameter sweeps
python db_cli.py show latest --config # one run in full, with the config it used
```

Measurements are filtered with the results command. Each filter is optional and they compose, so the question a user actually has — which settings answered fast enough while still being accurate enough — is a single command rather than a spreadsheet sort:

```bash
python db_cli.py results --min-recall 0.95 --max-latency 1.0 --sort latency
python db_cli.py results --run latest --method HNSW
python db_cli.py results --n-vectors 100000 --sort recall
python db_cli.py tuning --method IVF --min-recall 0.95 # cheapest setting that qualifies
```

Results can be sorted by scale, latency, recall, memory, or build time with --sort, exported to CSV for a report or a spreadsheet, and for anything the fixed commands do not cover there is read-only SQL. Writes are refused there deliberately: a recorded measurement is evidence, and evidence should not be editable from a convenience tool.

```bash
python db_cli.py export results --run latest --out results/for_report.csv
python db_cli.py summary # per-method rollup across every run
python db_cli.py sql "SELECT method, AVG(recall_at_k) FROM measurement GROUP BY method"
```

Four saved views cover the common questions without your having to write the joins: v_results is every measurement with its run context attached, v_method_summary is a per-method rollup across all completed runs, v_scalability is the most recent sweep, and v_tuning_curve is the parameter sweep ordered by knob value. If you prefer a graphical client, the database is an ordinary SQLite file and any SQLite browser will open it.

## 4. Troubleshooting & Support

### 4.1 Common errors

The codes in the first column are references into this guide, so a teammate can be pointed at UG-04 rather than having a stack trace pasted at them. The second column is the text you will actually see.

| Code | What you see | Cause and fix |
|---|---|---|
| UG-01 | ModuleNotFoundError: No module named 'faiss' | Dependencies were not installed, or were installed into a different interpreter. Run pip install -r requirements.txt, and check that python and pip refer to the same environment. |
| UG-02 | ImportError: sentence-transformers is required for the 'sbert' embedder | The config asks for the SBERT backend but the optional package is absent. Either pip install sentence-transformers, or set embedding.backend to hashing to run offline. |
| UG-03 | OSError / connection error while loading all-MiniLM-L6-v2 | The one-time model download could not reach Hugging Face. Check the network or the proxy; if the machine is air-gapped, switch to the hashing backend, which needs no download. |
| UG-04 | FileNotFoundError: configs/….yaml | The config path is relative to the project folder. Run the command from the folder containing run_benchmark.py. |
| UG-05 | Results look empty, or the corpus is far smaller than expected | data.path points at a file that does not exist, so ingestion silently fell back to the built-in 44-sentence sample corpus. Check the path, or run fetch_dataset.py. |
| UG-06 | WARNING clustering … training set too small (FAISS) | IVF was asked for more clusters than the data can train. The framework clamps nlist automatically, so if this still appears, lower params.ivf.nlist in the config. |
| UG-07 | WARN in the log: recall below 0.95 | Not a crash. The approximate index is under-searching. Raise nprobe for IVF or ef_search for HNSW — run tune_parameters.py to see what each value costs before choosing. |
| UG-08 | MemoryError, or the machine begins swapping during a sweep | The largest scale does not fit in RAM. Reduce the entries in scalability.sizes, or lower data.dim. |
| UG-09 | sqlite3.IntegrityError on writing results | A schema constraint rejected an impossible value, such as a recall outside 0 to 1. This indicates a genuine bug in the evaluation path — please report it with the run id. |
| UG-10 | sqlite3.OperationalError: database is locked | Two runs are writing to the same database file concurrently. Run them one at a time, or give one a separate file with --db. |
| UG-11 | No runs recorded yet. Run a benchmark first. | A db_cli.py command asked for 'latest' before anything had been run. Run a benchmark, or check that --db points at the right file. |
| UG-12 | Two runs of the same config give different latencies | Expected. Timings vary with machine load; recall, memory, and build size do not. Compare those for reproducibility, and compare latency only between runs on the same host. |

### 4.2 Diagnostics

When a run misbehaves, three commands usually locate the problem before any code is read. First check whether anything failed at all, then look at where the failing run stopped, then confirm the run was configured the way you think it was — the database stores every run's configuration verbatim, so this is a fact rather than a recollection:

```bash
python db_cli.py runs --status failed
python db_cli.py logs --run <id> --level ERROR
python db_cli.py show <id> --config
```

If the installation itself is suspect rather than a particular run, the test suite is the fastest way to find out. It covers ingestion, embedding determinism, all three index families, the evaluation metrics, and the database constraints, and it runs in a few seconds:

```bash
pip install pytest
pytest
```

### 4.3 Contact and support

Questions are best directed to the member who owns the relevant layer; the ownership split is the same one used throughout the project, so the person who wrote a module is the person who supports it.

| Area | Contact | Email |
|---|---|---|
| Dataset, ingestion, embeddings | Devu Babu Sheeja | babushe@uwindsor.ca |
| Index construction, FAISS, tuning parameters | Ashwin Senthur Pandian | senthur@uwindsor.ca |
| Query engine, metrics, results database | Rishi G. Patel | patel3zc@uwindsor.ca |
| CLI, configurations, charts, repository | Nikhil Goud Nathi | nathin@uwindsor.ca |
| Course and assessment | Dr. Andreas S. Maniatis | — |

Defects and feature requests should be raised as issues on the project repository at github.com/Devz0201/COMP8157-Vector-Indexing, where they are tracked on the team's Hive board. A useful report includes the run id, the output of python db_cli.py show <id> --config, and the relevant log lines — with those three things the run can usually be reproduced exactly, because the configuration, the machine, and the library versions are all recorded alongside the result.

Further documentation lives in the repository: README.md is the project portal, data/DATASET.md records the dataset's provenance and licence, db/schema.sql is the annotated database schema, and the D.4.1 requirements, D.4.2 design, and D.4.4 deployment documents are under docs/.
