# D.4.4 Deployment Document

*Performance Evaluation of Vector Indexing Techniques for AI Search Systems*

> This is the Markdown mirror of **`D.4.4_Deployment_Document_Group4.docx`**, kept in the repository so the
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

## Section 1: Pre-Deployment & Environment Setup

### 1.1 System Overview

The system being deployed is the Vector Index Benchmarking Framework, version 1.1.0, the application described in the D.4.1 requirements and D.4.2 design documents. It is a command-line benchmarking tool, not a hosted service: a user invokes it once per experiment, it builds Flat, IVF, and HNSW indexes over the same embedded corpus, measures latency, recall, memory, and build time under identical conditions, and writes its output to disk before exiting. Nothing stays running between invocations, and no port is ever opened.

That shape determines what “deployment” means here. There is no server to provision, no load balancer, and no blue-green cutover; deploying the application means placing the repository on a machine, installing its dependencies, and confirming that a benchmark run reproduces the numbers in the report. The team therefore treats three target environments as first-class, and all three are verified before this document is signed off.

| Environment | Target | Purpose |
|---|---|---|
| Production (grading) | Google Colab, hosted notebook | The zero-installation path. Runtime → Run all reproduces every result in roughly ten minutes. |
| Staging | GitHub Codespaces, Linux container | Clean-room verification that a fresh clone runs with no local state carried over. |
| Development | Team members' local machines (Windows 11, macOS, Ubuntu) | Day-to-day development and the runs that produced the committed results. |

Version 1.1.0 is the D.4 submission build. It differs from the 1.0.0 build described at the D.3.1 midterm checkpoint in three respects: the HNSW index and the parameter-tuning sweep are complete, a SQLite results database persists every run alongside the CSV artefacts, and the automated test suite and continuous-integration workflow were added.

### 1.2 Prerequisites

The application is CPU-only by design, as agreed at the proposal stage, so there is no GPU requirement and no CUDA toolchain to install. The hardware floor is set by the largest experiment rather than by the software: the 100,000-vector scalability sweep holds three indexes and their source vectors in memory at once, which peaks a little above 1 GB.

| Requirement | Minimum | Used for the reported results |
|---|---|---|
| CPU | 2 cores, x86-64 or ARM64 | 12-core x86-64 |
| RAM | 4 GB | 16 GB |
| Free disk | 2 GB (dependencies, model cache, results) | — |
| Operating system | Windows 10+, macOS 12+, or any current Linux | Windows 11 26200 |
| Python | 3.10 or newer | 3.13.1 |
| Network | Only for the one-time model and dataset downloads | — |

Two software prerequisites deserve a note. FAISS is installed from the faiss-cpu wheel, which is prebuilt for every supported platform, so no C++ compiler or BLAS configuration is needed on the target machine. Sentence-BERT is optional and installed separately: it pulls in PyTorch and downloads the all-MiniLM-L6-v2 model on first use, so an environment without it can still run the entire pipeline through the offline hashing and synthetic embedding backends, which exercise the identical indexing, query, and evaluation code.

On SSL certificates: the application terminates no TLS of its own, because it serves no traffic. TLS matters at exactly two points, both of them outbound and both handled by the operating system's certificate store — the Hugging Face model download and the Harvard Dataverse dataset download, each fetched over HTTPS. On a machine with an outdated root store or behind a TLS-inspecting corporate proxy those two fetches are the only steps that can fail; the rest of the pipeline is offline. The mitigation is to run the two downloads once on a machine with working certificates and copy the resulting data/headlines.txt and model cache across, after which the application is fully air-gapped.

### 1.3 Access & Roles

Deployment responsibilities follow the same four-lane ownership model used throughout the project, so the person who verifies a layer after deployment is the person who wrote it. Sign-off means that member has personally run the checks in their row on the target environment and seen them pass.

| Member | Deployment responsibility | Sign-off | Date |
|---|---|---|---|
| Devu Babu Sheeja | Dataset fetch and preparation; embedding backend availability; model cache | Approved | Jul 28, 2026 |
| Ashwin Senthur Pandian | FAISS installation and index construction across all three families | Approved | Jul 28, 2026 |
| Rishi G. Patel | Query engine, evaluation metrics, and results-database integrity | Approved | Jul 29, 2026 |
| Nikhil Goud Nathi | Repository, CLI entry points, Colab and Codespaces environments, CI | Approved | Jul 29, 2026 |

Repository access is held by all four members with write permission on github.com/Devz0201/COMP8157-Vector-Indexing; the course instructor and graders have read access, which is sufficient to clone, open a Codespace, or launch the Colab notebook. No member holds credentials the others do not, and the application itself stores no secrets, API keys, or tokens — both external downloads are unauthenticated public endpoints.

## Section 2: Execution & Post-Deployment Log

### 2.1 Step-by-Step Actions

The following is the exact command sequence executed on the production and staging targets. It is reproduced verbatim rather than summarised, because a deployment procedure that has been paraphrased is a procedure that has not been tested.

Step 1 — obtain the application and install its dependencies:

```bash
git clone https://github.com/Devz0201/COMP8157-Vector-Indexing.git
cd COMP8157-Vector-Indexing
pip install -r requirements.txt # faiss-cpu, numpy, pandas, matplotlib,
# scikit-learn, PyYAML (all prebuilt wheels)
pip install sentence-transformers # optional: the real Sentence-BERT path
```

Step 2 — create the results database. The schema in db/schema.sql is applied automatically by any run, so this command is a verification step rather than a prerequisite; it prints the tables and views it created.

```bash
python db_cli.py init
```

Step 3 — confirm the dataset is present. The repository ships the prepared 20,000-headline corpus at data/headlines.txt, so this step is normally a no-op. It is only needed to rebuild the corpus from source or to change the sample size:

```bash
python fetch_dataset.py --n 20000 # Harvard Dataverse, doi:10.7910/DVN/SYBGZL, CC0
# no account, no API key; ~61 MB one-time download
```

Step 4 — run the experiments. Either invoke the three configurations individually, or use the bundled launcher, which runs all three in the order the report presents them and then prints the database summary:

```bash
./run_all.sh # Linux / macOS (--offline skips the SBERT steps)
run_all.bat # Windows
# or, individually:
python run_benchmark.py --config configs/scalability.yaml
python run_benchmark.py --config configs/text_sbert.yaml
python tune_parameters.py --config configs/tuning.yaml
```

Step 5 — configuration updates applied during this deployment. One change was made to a committed configuration file, and it is recorded here rather than left in the commit history alone because it changes a reported result. In configs/text_sbert.yaml, IVF's nprobe was raised from 8 to 32; the reasoning is in the incident log below.

No environment variables need to be set. The application sets HF_HUB_VERBOSITY and TRANSFORMERS_VERBOSITY itself before importing the Hugging Face libraries, purely to keep benchmark output readable, and reads no configuration from the environment otherwise — every experimental parameter lives in the YAML file, which is the reproducibility requirement from D.4.1 enforced in practice.

### 2.2 Verification & Health Checks

Deployment is considered successful only when the following checks pass on the target machine. They are ordered cheapest-first, so a broken environment fails in seconds rather than after a ten-minute benchmark.

| # | Check | Command | Pass condition |
|---|---|---|---|
| V-1 | Dependencies import | python -c "import faiss, numpy, pandas, yaml" | Exits 0 and prints the FAISS version |
| V-2 | Test suite | pytest | 43 passed |
| V-3 | Lint | ruff check . | No findings |
| V-4 | End-to-end smoke run | python run_benchmark.py --config configs/ci_smoke.yaml | Exits 0; CSV and PNGs written under results/ci/ |
| V-5 | Database write path | python db_cli.py runs | The smoke run appears with status completed |
| V-6 | Exact-search sanity | python db_cli.py results --run latest --method Flat | recall_at_k is exactly 1.0 (AC-2) |
| V-7 | Approximate-search accuracy | python db_cli.py results --run latest --min-recall 0.95 | IVF and HNSW both appear (AC-3) |
| V-8 | Error monitoring | python db_cli.py logs --run latest --level ERROR | Returns no rows |

Error monitoring is built into the application rather than bolted on beside it. Every run writes its execution log into the run_log table of the same database that holds its results, tagged by severity and by pipeline stage, so a failure and the partial results it produced are always in one place. A run that raises an exception is recorded with status 'failed' and an ERROR row before the traceback is re-raised, which means V-8 is a genuine health check across every run ever executed on the machine and not merely a check of the last one.

The same checks run automatically on every push through the GitHub Actions workflow in .github/workflows/ci.yml, across Ubuntu and Windows on Python 3.10, 3.11, and 3.13. That matrix exists precisely because the project's central claim is reproducibility: a framework that only runs on the machine of whoever wrote the module would not support that claim, so every push is deployed to six clean machines and verified there.

### 2.3 Incident & Rollback Log

Timestamps below are UTC and are taken from the run table of db/vecbench_results.db, so each entry can be checked against the database rather than taken on trust.

| Time (UTC) | Severity | Incident | Resolution |
|---|---|---|---|
| 2026-07-29 16:09 | Info | Scalability sweep deployed and run, 1k–100k, run_id 1. | Completed. HNSW recall decay at fixed ef_search logged as WARN — expected and reported as a finding. |
| 2026-07-29 16:10 | Major | Run 2 (real SBERT, 20k headlines) returned IVF recall@10 = 0.839, below the 0.95 threshold in acceptance criterion AC-3. | See analysis below. Not rolled back — retained as evidence. |
| 2026-07-29 16:11 | Info | Diagnostic parameter sweep executed, run_id 3. | Identified nprobe = 32 as the first value clearing 0.95 on this corpus. |
| 2026-07-29 16:13 | Resolved | configs/text_sbert.yaml updated, nprobe 8 → 32; benchmark re-run as run_id 4. | IVF recall@10 = 0.9625, HNSW = 0.9950. AC-3 passes. Verified by V-7. |

The one major incident is worth setting out in full, because it is also the clearest demonstration of why the framework exists. The IVF defaults — nlist = 100, nprobe = 8 — reach recall 1.000 on the synthetic clustered vectors used for the scalability sweep, and had done so in every run since the midterm checkpoint. On real Sentence-BERT embeddings of news headlines the same settings reached only 0.839. The cause is not a defect: synthetic Gaussian clusters are well separated, so searching eight of a hundred cells finds nearly every true neighbour, whereas real headline embeddings overlap far more and the true neighbours of a query are spread across many more cells. Tuning on synthetic data and deploying against real data is exactly the mistake the project set out to expose in other people's benchmarks, and the framework caught the team making it.

The resolution was to run the parameter sweep the framework already provides, read the IVF curve on the real corpus (0.505 at nprobe 1, rising through 0.839 at 8 and 0.910 at 16 to 0.963 at 32), and set the deployed default to the cheapest value that clears the acceptance threshold. Both runs are retained in the database, so the before-and-after comparison is a single query rather than a claim.

Rollback procedure. Because the application is a stateless command-line tool, reverting a deployment is reverting a commit — git checkout of the previous tag restores the previous behaviour completely, since no schema migration, no service restart, and no data conversion is involved. Results are append-only and immutable in practice: a superseded run is never edited or deleted, so rolling the code back does not destroy the evidence of what the newer version measured. If a run must be removed — an aborted or mis-configured execution — deleting its row cascades to its measurements, tuning points, and log lines through the foreign keys in db/schema.sql, leaving no orphaned rows:

```bash
git checkout v1.0.0 # revert the application
python db_cli.py sql "SELECT run_id, status FROM run" # identify the run
sqlite3 db/vecbench_results.db "DELETE FROM run WHERE run_id = 2;"
```

No rollback was required during this deployment. The single major incident was resolved forward, by tuning a parameter and re-running, rather than by reverting.

### 2.4 Post-Deployment Status

All three target environments are deployed and verified as of July 29, 2026. Checks V-1 through V-8 pass on each; the test suite reports 43 passed; the results database holds four completed runs and no failed ones; and every figure and table in the D.5 report is generated from the artefacts under results/ produced by those runs. The application is released for grading.
