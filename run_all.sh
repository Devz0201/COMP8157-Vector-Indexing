#!/usr/bin/env bash
# =====================================================================
# run_all.sh -- one-command reproduction of every result in the report
# Vector Index Benchmarking Framework, COMP.8157 Group 4 (Linux / macOS)
# =====================================================================
#
# Runs the three experiments in the order the report presents them, then
# prints the database summary. Everything it produces (CSV tables, PNG
# charts, and the run rows in db/vecbench_results.db) is what the report
# and the presentation are built from.
#
#   ./run_all.sh              full reproduction, including the SBERT run
#   ./run_all.sh --offline    skip anything needing a model download
#
# The Windows equivalent is run_all.bat.

set -euo pipefail
cd "$(dirname "$0")"

OFFLINE=0
[[ "${1:-}" == "--offline" ]] && OFFLINE=1

banner() {
    echo ""
    echo "======================================================================"
    echo "  $1"
    echo "======================================================================"
}

banner "0/4  Environment check"
python --version
python -c "import faiss, numpy, pandas, matplotlib, yaml; print('core dependencies OK, faiss', faiss.__version__)"
python db_cli.py init

banner "1/4  Scalability sweep (synthetic vectors, 1k to 100k, offline)"
python run_benchmark.py --config configs/scalability.yaml \
    --notes "run_all.sh scalability sweep"

if [[ "$OFFLINE" -eq 1 ]]; then
    echo ""
    echo "--offline given: skipping the Sentence-BERT experiments (steps 2 and 3)."
else
    if [[ ! -f data/headlines.txt ]]; then
        banner "Dataset missing -- fetching A Million News Headlines"
        python fetch_dataset.py --n 20000
    fi

    banner "2/4  Real-data benchmark (20,000 news headlines, Sentence-BERT)"
    python run_benchmark.py --config configs/text_sbert.yaml \
        --notes "run_all.sh SBERT benchmark"

    banner "3/4  Parameter-tuning sweep (IVF nprobe, HNSW efSearch)"
    python tune_parameters.py --config configs/tuning.yaml \
        --notes "run_all.sh tuning sweep"
fi

banner "4/4  Results database summary"
python db_cli.py runs
python db_cli.py summary

echo ""
echo "Done. Tables and charts are under results/; every run is recorded in"
echo "db/vecbench_results.db (browse it with:  python db_cli.py show latest)."
