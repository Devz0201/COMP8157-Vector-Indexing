"""Vector Indexing Benchmarking Framework.

A workload-consistent benchmark comparing Flat, IVF, and HNSW vector indexes
under identical conditions, measuring latency, recall, memory, build time, and
scalability.
"""

from .benchmark import run_scalability, run_single_scale  # noqa: F401
from .results_db import ResultsDB  # noqa: F401

__all__ = ["run_single_scale", "run_scalability", "ResultsDB"]
__version__ = "1.1.0"
