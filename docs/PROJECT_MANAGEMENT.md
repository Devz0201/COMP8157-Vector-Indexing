# Project management

**Performance Evaluation of Vector Indexing Techniques for AI Search Systems**
COMP.8157 Advanced Database Topics, 2026S — University of Windsor — Group 4

| | |
|---|---|
| **Project management tool** | Hive — [board](https://app.hive.com/workspace/JbYCPPC6L8DCGyzKG?actionViewId=W4yJq7vXwthAk2DqF&tabId=aprxRosGk2vxPJixC&projectId=48q28Z9gpJfRKWDJs) |
| **Code repository** | [github.com/Devz0201/COMP8157-Vector-Indexing](https://github.com/Devz0201/COMP8157-Vector-Indexing) |
| **Contribution log** | [`docs/contributions/Team_Contribution_Log.pdf`](contributions/Team_Contribution_Log.pdf) — weekly, per member |
| **Contribution form** | [`docs/contributions/D3_1_1_Team_Contribution_Form_Group4.docx`](contributions/) |

---

## How the work was organised

The team used a **four-lane ownership model** rather than a shared pool of
tasks. Each member owns a disjoint set of code modules, which means two people
never edit the same file, merge conflicts are close to nonexistent, and the
commit history is itself the evidence of who did what — no separate claim has
to be taken on trust.

The lanes follow the five-layer architecture, so a lane boundary is also an
interface boundary. That is what let the team report honest, granular progress
at the D.3.1 midterm checkpoint ("Flat and IVF complete, HNSW in progress")
without any finished layer depending on something still unfinished.

| Lane | Owner | GitHub | Modules owned |
|---|---|---|---|
| Data & Embeddings | Devu Babu Sheeja | `Devz0201` | `data_ingestion.py`, `prepare_headlines.py`, `embeddings.py`, `fetch_dataset.py` |
| Vector Indexing | Ashwin Senthur Pandian | `ashwin-0707` | `indexing.py` (Flat, IVF, HNSW) |
| Query & Evaluation | Rishi G. Patel | `Rishihi29` | `query_engine.py`, `benchmark.py`, `results_db.py`, `db/schema.sql` |
| Experiments, Tuning & Visualization | Nikhil Goud Nathi | `nikhilnathi2003` | `run_benchmark.py`, `tune_parameters.py`, `visualization.py`, `db_cli.py`, `configs/` |

Shared across all four members: the proposal and report writing, the literature
review, integration testing and code review, reproducibility checks, and the
presentation.

---

## Phases

The proposal's five phases map onto the deliverable deadlines as follows.

| Phase | Work | Delivered by |
|---|---|---|
| **P1** Data & Preprocessing | Dataset selection and justification, ingestion, cleaning, segmentation | D.2.1 → D.3.1 |
| **P2** Embedding | Sentence-BERT backend, offline backends, determinism | D.3.1 |
| **P3** Index Construction | Flat and IVF (complete at D.3.1), HNSW and adaptive `nlist` | D.3.1 → D.4 |
| **P4** Benchmarking & Evaluation | Query engine, recall@K, scalability sweep, parameter tuning, results database | D.4 |
| **P5** Analysis & Documentation | Charts, requirement/design/deployment/user documents, report, presentation | D.4 → D.6 |

---

## Board structure

The Hive board carries one column per state (`Backlog` → `In
progress` → `In review` → `Done`) and one label per lane, so the board can be
filtered down to a single member's work at a glance. Issues are written at the
granularity of the sixteen-task breakdown from the D.3.1 midterm report, and
each issue is closed by the commit that implements it, which keeps the board
and the repository in step automatically instead of by discipline.

Two conventions the team settled on early and kept:

- **Nothing is pushed until it runs.** At the midterm checkpoint the HNSW index
  and the tuning script existed locally but were not in the repository, because
  they were not yet tested. The status report said so explicitly. The
  alternative — pushing untested work and reporting it as done — would have
  made the repository a worse record of progress, not a better one.
- **The config, not the code, records an experiment.** Every result in the
  report can be traced to a YAML file committed next to the code that produced
  it, and since D.4 also to a row in the results database that stores that
  config verbatim alongside the machine it ran on.

---

## Milestones

| Deliverable | Due | Status |
|---|---|---|
| D.1 Project pre-proposal | May 17, 2026 | Complete |
| D.2.1 Full proposal + presentation | May 31 / Jun 2, 2026 | Complete |
| D.3.1 Midterm status report | Jun 28, 2026 | Complete |
| D.3.2 Individual contribution reports | Jun 28, 2026 | Complete |
| D.4 Final demo — application, code, documentation | Jul 21, 2026 | Complete |
| D.4.1 User Requirements and Analysis | Jul 21, 2026 | Complete |
| D.4.2 Design document | Jul 21, 2026 | Complete |
| D.4.3 Application, dataset, schema | Jul 31, 2026 | Complete |
| D.4.4 Deployment document | Jul 31, 2026 | Complete |
| D.4.5 User Guide | Jul 31, 2026 | Complete |
| D.5 Final project report | Jul 31, 2026 | Complete |
| D.6.1 Presentation deck | Jul 31, 2026 | In progress |
| D.6.2 In-class presentation | Aug 4, 2026 | Scheduled |

---

## Risk register

Carried forward from the D.3.1 midterm report, with the outcome of each.

| Risk | Impact | Response | Outcome |
|---|---|---|---|
| FAISS k-means warns or fails at small dataset sizes (needs ~39 training points per cluster) | Scalability sweep unusable at its low end | Clamp `nlist` to `n // 39` before training | **Resolved** — covered by `test_ivf_clamps_nlist_on_small_datasets` |
| Sentence-BERT model download unavailable on a grading machine | Real-data benchmark cannot be run | Ship offline `hashing` and `synthetic` backends exercising the same downstream code | **Resolved** — CI runs the full pipeline with no model download |
| Timing noise makes methods indistinguishable | Results not defensible | Warm-up queries discarded, per-query timing, p50/p95 reported | **Resolved** |
| Results scattered across per-run CSV files become impossible to compare | Cross-run analysis done by hand and error-prone | Persist every run to a SQLite database with its config and environment | **Resolved** at D.4 |
| HNSW recall decays as N grows at fixed `ef_search` | Could be misread as an HNSW defect | Documented as a tuning requirement; quantified by the parameter sweep | **Resolved** — reported as a finding, not a bug |
