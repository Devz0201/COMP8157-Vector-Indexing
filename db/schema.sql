-- =====================================================================
-- Vector Index Benchmarking Framework -- Results Database Schema
-- COMP.8157 Advanced Database Topics, 2026S -- Group 4
-- Engine: SQLite 3 (bundled with Python, no server to install)
-- =====================================================================
--
-- Why a database at all, when the framework already writes CSV files?
--
-- A CSV is a snapshot of one run. Once the team started sweeping dataset
-- sizes and tuning parameters, the interesting questions stopped being
-- "what did this run report" and became "how does HNSW recall at 100k
-- compare with the run we did last week, on the same machine, at the same
-- efSearch". Answering that from a pile of CSVs means re-reading files by
-- hand; answering it from one table is a WHERE clause. So every run is
-- also appended here, and the CSVs remain as the per-run artefact.
--
-- The schema is deliberately small: one row per benchmark run, one row per
-- (method, dataset size) measurement inside that run, one row per tuning
-- sweep point, and one row per log line. Everything a chart or a report
-- needs can be reconstructed from these four tables.
--
-- Apply with:   sqlite3 db/vecbench_results.db < db/schema.sql
-- or:           python db_cli.py init
-- =====================================================================

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------
-- run -- one row per invocation of run_benchmark.py or tune_parameters.py
--
-- The environment columns (hostname, platform, cpu_count, ...) are not
-- decoration. The project's central claim is that measured differences
-- come from the index structure and nothing else, which only holds if two
-- rows being compared were produced on the same machine and the same
-- library versions. Recording them makes that checkable after the fact
-- instead of relying on memory.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS run (
    run_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at         TEXT    NOT NULL,           -- ISO-8601 UTC
    finished_at        TEXT,                       -- NULL while still running
    status             TEXT    NOT NULL DEFAULT 'running'
                               CHECK (status IN ('running', 'completed', 'failed')),
    experiment         TEXT    NOT NULL
                               CHECK (experiment IN ('single', 'scalability', 'tuning')),
    config_path        TEXT    NOT NULL,           -- e.g. configs/text_sbert.yaml
    config_yaml        TEXT    NOT NULL,           -- the full config, verbatim
    dataset            TEXT,                       -- corpus path, or 'synthetic'
    embedding_backend  TEXT,                       -- sbert | hashing | synthetic
    embedding_model    TEXT,                       -- e.g. all-MiniLM-L6-v2
    k                  INTEGER,                    -- top-K used for retrieval
    n_queries          INTEGER,
    hostname           TEXT,
    platform           TEXT,
    python_version     TEXT,
    faiss_version      TEXT,
    cpu_count          INTEGER,
    notes              TEXT
);

-- ---------------------------------------------------------------------
-- measurement -- one row per (run, method, dataset size)
--
-- This is the MethodMetrics dataclass persisted. A 'single' run produces
-- three rows (Flat/IVF/HNSW); a five-size scalability sweep produces
-- fifteen. The UNIQUE constraint encodes the fact that a run measures each
-- method exactly once per scale, so a re-run cannot silently double up.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS measurement (
    measurement_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           INTEGER NOT NULL
                             REFERENCES run(run_id) ON DELETE CASCADE,
    method           TEXT    NOT NULL
                             CHECK (method IN ('Flat', 'IVF', 'HNSW')),
    label            TEXT    NOT NULL,   -- method + its parameters, e.g. IVF(nlist=128,nprobe=8)
    n_vectors        INTEGER NOT NULL CHECK (n_vectors > 0),
    k                INTEGER NOT NULL CHECK (k > 0),
    build_time_s     REAL    NOT NULL CHECK (build_time_s >= 0),
    memory_mb        REAL    NOT NULL CHECK (memory_mb    >= 0),
    latency_mean_ms  REAL    NOT NULL CHECK (latency_mean_ms >= 0),
    latency_p50_ms   REAL    NOT NULL,
    latency_p95_ms   REAL    NOT NULL,
    qps              REAL    NOT NULL,
    -- recall is a fraction: anything outside [0,1] means the evaluator is broken
    recall_at_k      REAL    NOT NULL CHECK (recall_at_k BETWEEN 0.0 AND 1.0),
    UNIQUE (run_id, method, n_vectors)
);

-- ---------------------------------------------------------------------
-- tuning_point -- one row per parameter setting tried in a tuning sweep
--
-- Kept separate from `measurement` rather than folded into it: a tuning
-- point varies a query-time knob on an index that was built once, so it
-- has no build time and no memory footprint of its own. Forcing it into
-- the measurement table would mean three permanently-NULL columns.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tuning_point (
    tuning_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
    method      TEXT    NOT NULL CHECK (method IN ('IVF', 'HNSW')),
    knob        TEXT    NOT NULL CHECK (knob IN ('nprobe', 'efSearch')),
    value       INTEGER NOT NULL CHECK (value > 0),
    recall      REAL    NOT NULL CHECK (recall BETWEEN 0.0 AND 1.0),
    latency_ms  REAL    NOT NULL CHECK (latency_ms >= 0),
    UNIQUE (run_id, method, knob, value)
);

-- ---------------------------------------------------------------------
-- run_log -- the execution log of every run, kept with the results
--
-- The console output of a run scrolls away; this keeps it. It is what the
-- User Guide's "viewing logs" and "filtering and searching data" features
-- read from, and it is the first place to look when a run fails, because
-- the ERROR line sits in the same database as the partial results it
-- produced.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS run_log (
    log_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   INTEGER NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
    ts       TEXT    NOT NULL,          -- ISO-8601 UTC
    level    TEXT    NOT NULL CHECK (level IN ('DEBUG', 'INFO', 'WARN', 'ERROR')),
    stage    TEXT,                      -- ingest | embed | index | query | evaluate | report
    message  TEXT    NOT NULL
);

-- ---------------------------------------------------------------------
-- Indexes
--
-- Every query the CLI issues is either "everything for this run" or
-- "this method across runs", so those are the two access paths indexed.
-- The log index is (run_id, level) because filtering logs by severity
-- within a run is by far the most common troubleshooting query.
-- ---------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_measurement_run    ON measurement (run_id);
CREATE INDEX IF NOT EXISTS idx_measurement_method ON measurement (method, n_vectors);
CREATE INDEX IF NOT EXISTS idx_tuning_run         ON tuning_point (run_id, method);
CREATE INDEX IF NOT EXISTS idx_log_run_level      ON run_log (run_id, level);
CREATE INDEX IF NOT EXISTS idx_run_experiment     ON run (experiment, started_at);

-- =====================================================================
-- Views -- the questions we actually ask, saved as queries
-- =====================================================================

-- Every measurement with its run context attached, so a reader never has
-- to write the join by hand. This is the view the CLI's `results` command
-- filters over.
CREATE VIEW IF NOT EXISTS v_results AS
SELECT
    m.measurement_id,
    m.run_id,
    r.experiment,
    r.started_at,
    r.dataset,
    r.embedding_backend,
    m.method,
    m.label,
    m.n_vectors,
    m.k,
    m.latency_mean_ms,
    m.latency_p50_ms,
    m.latency_p95_ms,
    m.qps,
    m.recall_at_k,
    m.memory_mb,
    m.build_time_s
FROM measurement m
JOIN run r ON r.run_id = m.run_id;

-- Per-method rollup across every completed run. Answers "on average, how
-- do the three methods compare" without re-reading a single CSV.
CREATE VIEW IF NOT EXISTS v_method_summary AS
SELECT
    m.method,
    COUNT(*)                     AS n_measurements,
    MIN(m.n_vectors)             AS min_scale,
    MAX(m.n_vectors)             AS max_scale,
    ROUND(AVG(m.latency_mean_ms), 5) AS avg_latency_ms,
    ROUND(AVG(m.recall_at_k),     4) AS avg_recall,
    ROUND(AVG(m.memory_mb),       3) AS avg_memory_mb,
    ROUND(AVG(m.build_time_s),    4) AS avg_build_s
FROM measurement m
JOIN run r ON r.run_id = m.run_id
WHERE r.status = 'completed'
GROUP BY m.method;

-- The scalability trend, one row per (method, scale), taking the most
-- recent completed sweep for each so repeated runs don't stack up.
CREATE VIEW IF NOT EXISTS v_scalability AS
SELECT
    m.method,
    m.n_vectors,
    m.latency_mean_ms,
    m.recall_at_k,
    m.memory_mb,
    m.build_time_s,
    m.run_id
FROM measurement m
JOIN run r ON r.run_id = m.run_id
WHERE r.experiment = 'scalability'
  AND r.status = 'completed'
  AND r.run_id = (SELECT MAX(run_id) FROM run
                  WHERE experiment = 'scalability' AND status = 'completed')
ORDER BY m.n_vectors, m.method;

-- The cheapest setting that still hits a given recall bar is the practical
-- output of a tuning sweep, so it gets its own view: for each method, the
-- knob values ordered by cost, with recall alongside.
CREATE VIEW IF NOT EXISTS v_tuning_curve AS
SELECT
    t.run_id,
    t.method,
    t.knob,
    t.value,
    t.recall,
    t.latency_ms
FROM tuning_point t
JOIN run r ON r.run_id = t.run_id
WHERE r.status = 'completed'
ORDER BY t.method, t.value;
