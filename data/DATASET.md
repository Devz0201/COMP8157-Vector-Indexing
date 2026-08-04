# Dataset metadata

The benchmark is run over two kinds of input: one real text corpus, and a
synthetic vector generator used for the large scalability sweeps. Both are
described below, along with exactly how the committed input file was produced.

---

## 1. A Million News Headlines (the real corpus)

| | |
|---|---|
| **Name** | A Million News Headlines |
| **Publisher** | Rohit Kulkarni, via Harvard Dataverse |
| **Persistent ID** | [doi:10.7910/DVN/SYBGZL](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SYBGZL) |
| **Mirror** | [kaggle.com/datasets/therohk/million-headlines](https://www.kaggle.com/datasets/therohk/million-headlines) (requires a free account) |
| **Licence** | CC0 1.0 Universal — public domain dedication, no attribution required |
| **Downloaded from** | Harvard Dataverse, original-format CSV, via `fetch_dataset.py` |
| **Raw size** | ~61 MB, 1,244,184 rows |
| **Coverage** | ABC (Australian Broadcasting Corporation) news headlines, Feb 2003 – Dec 2021 |
| **Columns** | `publish_date` (YYYYMMDD integer), `headline_text` (lower-cased headline string) |
| **Contains personal data?** | No. Published news headlines only. |

### Why this dataset

Headlines are short, self-contained, and topically diverse, which is close to
the shape of the documents a real semantic-search backend indexes. There is
enough volume to scale the benchmark well past 100,000 vectors, the licence
places no restriction on redistribution, and the Dataverse copy can be fetched
without an account — so every number in the report can be reproduced by anyone
with the repository and a network connection.

### How `headlines.txt` was produced

`data/headlines.txt` is committed to the repository so the real-data benchmark
runs straight after a clone. It is a deterministic sample of the raw CSV:

```bash
python fetch_dataset.py --n 20000 --seed 42
# equivalently, from an already-downloaded CSV:
python prepare_headlines.py --csv abcnews-date-text.csv --out data/headlines.txt --n 20000 --seed 42
```

The preparation step reads the `headline_text` column, drops duplicate
headlines (keeping first occurrence), samples down to `--n` with a fixed seed,
and writes one headline per line.

| Stage | Rows |
|---|---|
| Raw CSV | 1,244,184 |
| After de-duplication | 1,213,003 |
| Committed sample (`--n 20000 --seed 42`) | 20,000 |

Re-running the command with the same seed reproduces the committed file
exactly. The raw CSV itself is **not** committed — it is 61 MB of data that
`fetch_dataset.py` can retrieve on demand, and keeping it out of version
control keeps the repository small enough to clone quickly.

### Derived data in the pipeline

`headlines.txt` is not the end of the chain. The ingestion layer cleans and
segments each line, holds out 200 of them as the query set, and the embedding
layer turns the remainder into 384-dimensional Sentence-BERT vectors
(`all-MiniLM-L6-v2`). Those vectors are what the indexes are actually built
over. Nothing is cached between runs, so a re-run re-derives everything from
this file.

---

## 2. Synthetic clustered vectors (the scalability input)

The 1,000 → 100,000 scalability sweep does **not** use the headlines corpus. It
generates vectors directly, via `SyntheticEmbedder` in `src/embeddings.py`:
64 Gaussian cluster centres in 384 dimensions, points drawn around them with
σ = 0.15, then L2-normalised. Queries are made by perturbing randomly chosen
base vectors (σ = 0.05) so every query has genuine near neighbours in the
index.

This is generated, not downloaded, so there is no licence or provenance
question attached to it. It is used for the scalability sweep for two reasons:
the dataset size becomes a free parameter rather than something bounded by the
corpus, and embedding 100,000 headlines with Sentence-BERT at every step of the
sweep would dominate the runtime without changing what the sweep measures — how
the *index* behaves as N grows. Cluster structure is included deliberately,
because uniformly random vectors in high dimensions are close to equidistant
and would make IVF's k-means partitioning look artificially useless.

---

## 3. Built-in sample corpus

`data/sample_corpus.txt` and the larger `_SAMPLE_CORPUS` constant inside
`src/data_ingestion.py` are 44 hand-written sentences across mixed themes
(machine learning, biology, finance, history, sport, astronomy). They exist so
the entire pipeline can be demonstrated with zero downloads — if `data.path` is
missing or unset, ingestion falls back to them. They are the team's own text,
not sourced from anywhere.
