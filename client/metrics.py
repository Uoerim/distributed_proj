"""
client/metrics.py
=================
Metrics collector and reporter for the Distributed LLM Load Balancing System.

Collects per-run results and produces:
  - Per-strategy comparison table (round_robin / least_connections / load_aware)
  - P50 / P95 / P99 latency breakdown
  - Throughput and failure rate per strategy
  - Per-worker request distribution
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    """Holds the results of a single load-test run."""
    strategy:      str
    num_users:     int
    elapsed:       float
    success_count: int
    fail_count:    int
    throughput:    float
    avg_latency:   float
    p50_latency:   float
    p95_latency:   float
    p99_latency:   float
    worker_counts: Dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def compute_percentiles(latencies: List[float]) -> dict:
    """Return avg, p50, p95, p99 from a raw latency list."""
    if not latencies:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    s = sorted(latencies)
    n = len(s)

    def _pct(p: float) -> float:
        idx = max(0, int(n * p / 100) - 1)
        return s[idx]

    return {
        "avg": statistics.mean(s),
        "p50": _pct(50),
        "p95": _pct(95),
        "p99": _pct(99),
    }


# ---------------------------------------------------------------------------
# Safe print helper (Windows cp1252 compatible)
# ---------------------------------------------------------------------------

def _safe_print(text: str) -> None:
    """Print text safely on Windows terminals that don't support Unicode."""
    print(text.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Accumulates results across multiple strategy runs and prints reports."""

    def __init__(self) -> None:
        self._runs: List[RunMetrics] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, strategy: str, metrics: dict) -> None:
        """Store the result of one run_load_test() call."""
        run = RunMetrics(
            strategy      = strategy,
            num_users     = metrics.get("num_users", 0),
            elapsed       = metrics.get("elapsed", 0.0),
            success_count = metrics.get("success_count", 0),
            fail_count    = metrics.get("fail_count", 0),
            throughput    = metrics.get("throughput", 0.0),
            avg_latency   = metrics.get("avg_latency", 0.0),
            p50_latency   = metrics.get("p50_latency", 0.0),
            p95_latency   = metrics.get("p95_latency", 0.0),
            p99_latency   = metrics.get("p99_latency", 0.0),
            worker_counts = metrics.get("worker_counts", {}),
        )
        self._runs.append(run)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_comparison(self) -> None:
        """Print a formatted per-strategy comparison table to stdout."""
        if not self._runs:
            print("[MetricsCollector] No runs recorded yet.")
            return

        col = 20
        num = 10
        sep = "-" * 82

        print(f"\n{'=' * 82}")
        print(f"  STRATEGY COMPARISON TABLE")
        print(f"{'=' * 82}")
        print(
            f"  {'Strategy':<{col}} {'Users':>{num}} {'Throughput':>{num}} "
            f"{'Avg':>{num}} {'P50':>{num}} {'P95':>{num}} {'P99':>{num}} {'Failed':>{num}}"
        )
        print(f"  {sep}")

        for r in self._runs:
            fail_pct = (r.fail_count / r.num_users * 100) if r.num_users else 0
            print(
                f"  {r.strategy:<{col}} "
                f"{r.num_users:>{num}} "
                f"{r.throughput:>{num}.2f} "
                f"{r.avg_latency:>{num}.3f}s "
                f"{r.p50_latency:>{num}.3f}s "
                f"{r.p95_latency:>{num}.3f}s "
                f"{r.p99_latency:>{num}.3f}s "
                f"{r.fail_count:>{num}} ({fail_pct:.1f}%)"
            )

        print(f"{'=' * 82}")
        print(f"  Units: Throughput = req/s | Latency columns = seconds\n")

    def print_worker_distribution(self) -> None:
        """Print per-worker request distribution for each recorded run."""
        if not self._runs:
            return

        print(f"\n{'=' * 50}")
        print(f"  WORKER REQUEST DISTRIBUTION")
        print(f"{'=' * 50}")

        for r in self._runs:
            if not r.worker_counts:
                print(f"  [{r.strategy}]  no worker data available")
                continue

            print(f"\n  Strategy: {r.strategy}  |  Total successful: {r.success_count}")
            total = sum(r.worker_counts.values()) or 1
            for wid in sorted(r.worker_counts):
                count   = r.worker_counts[wid]
                bar_len = int(count / total * 30)
                bar     = "#" * bar_len          # ASCII-safe, no Unicode blocks
                print(f"    Worker {wid}: {count:>5} reqs  [{bar:<30}] ({count/total*100:.1f}%)")

        print(f"{'=' * 50}\n")

    def best_strategy(self) -> Optional[str]:
        """Return the strategy name with the highest throughput."""
        if not self._runs:
            return None
        best = max(self._runs, key=lambda r: r.throughput)
        return best.strategy