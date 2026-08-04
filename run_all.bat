@echo off
REM =====================================================================
REM run_all.bat -- one-command reproduction of every result in the report
REM Vector Index Benchmarking Framework, COMP.8157 Group 4 (Windows)
REM =====================================================================
REM
REM Runs the three experiments in the order the report presents them, then
REM prints the database summary. Everything it produces (CSV tables, PNG
REM charts, and the run rows in db\vecbench_results.db) is what the report
REM and the presentation are built from.
REM
REM   run_all.bat              full reproduction, including the SBERT run
REM   run_all.bat --offline    skip anything needing a model download
REM
REM The Linux / macOS equivalent is run_all.sh.

setlocal
cd /d "%~dp0"

set OFFLINE=0
if "%~1"=="--offline" set OFFLINE=1

echo.
echo ======================================================================
echo   0/4  Environment check
echo ======================================================================
python --version || goto :failed
python -c "import faiss, numpy, pandas, matplotlib, yaml; print('core dependencies OK, faiss', faiss.__version__)" || goto :failed
python db_cli.py init || goto :failed

echo.
echo ======================================================================
echo   1/4  Scalability sweep (synthetic vectors, 1k to 100k, offline)
echo ======================================================================
python run_benchmark.py --config configs/scalability.yaml --notes "run_all.bat scalability sweep" || goto :failed

if "%OFFLINE%"=="1" (
    echo.
    echo --offline given: skipping the Sentence-BERT experiments ^(steps 2 and 3^).
    goto :summary
)

if not exist "data\headlines.txt" (
    echo.
    echo ======================================================================
    echo   Dataset missing -- fetching A Million News Headlines
    echo ======================================================================
    python fetch_dataset.py --n 20000 || goto :failed
)

echo.
echo ======================================================================
echo   2/4  Real-data benchmark (20,000 news headlines, Sentence-BERT)
echo ======================================================================
python run_benchmark.py --config configs/text_sbert.yaml --notes "run_all.bat SBERT benchmark" || goto :failed

echo.
echo ======================================================================
echo   3/4  Parameter-tuning sweep (IVF nprobe, HNSW efSearch)
echo ======================================================================
python tune_parameters.py --config configs/tuning.yaml --notes "run_all.bat tuning sweep" || goto :failed

:summary
echo.
echo ======================================================================
echo   4/4  Results database summary
echo ======================================================================
python db_cli.py runs || goto :failed
python db_cli.py summary || goto :failed

echo.
echo Done. Tables and charts are under results\; every run is recorded in
echo db\vecbench_results.db (browse it with:  python db_cli.py show latest).
endlocal
exit /b 0

:failed
echo.
echo A step failed. See docs\USER_GUIDE.md, "Troubleshooting", for the error codes.
endlocal
exit /b 1
