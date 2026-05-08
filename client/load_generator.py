"""
client/load_generator.py
========================
Client Load Generator for the Distributed LLM Inference System.

Simulates concurrent users sending requests to the Scheduler,
modelling real-world traffic patterns for load testing and
performance evaluation.

Features
--------
* Configurable number of concurrent users (default 1000).
* Thread-based concurrency using ``threading.Thread``.
* Staggered request dispatch to avoid overwhelming ngrok tunnels.
* Optional semaphore-based max_concurrent cap.
* Per-request latency and success/failure logging.
* Aggregate summary with P50 / P95 / P99 latency percentiles.
* Optional failure simulation: kills a chosen worker mid-test, then
  optionally recovers it to verify fault-tolerance & task reassignment.
* Returns a metrics dict (consumed by MetricsCollector).
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

from common.models import Request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-user simulation
# ---------------------------------------------------------------------------

def simulate_user(
    scheduler,
    user_id: int,
    results: list,
    lock: threading.Lock,
    semaphore: Optional[threading.Semaphore] = None,
) -> None:
    """Simulate a single user sending one request."""
    ctx = semaphore if semaphore is not None else _NullContext()
    with ctx:
        request = Request(id=user_id, query=f"Query {user_id}")
        try:
            response = scheduler.handle_request(request)
            with lock:
                results.append((user_id, response))

            if response.success:
                logger.debug(
                    "[Client] User %d | Response %d | Latency: %.3fs",
                    user_id, response.id, response.latency,
                )
            else:
                logger.warning(
                    "[Client] User %d | FAILED | %s",
                    user_id, response.result[:60],
                )
        except Exception as exc:
            logger.error("[Client] User %d | Exception: %s", user_id, exc)


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Failure simulation
# ---------------------------------------------------------------------------

def _failure_simulation_thread(
    workers: list,
    kill_worker_id: int,
    kill_after_seconds: float,
    recover_after_seconds: Optional[float],
) -> None:
    """Background thread that kills (and optionally recovers) a worker."""
    time.sleep(kill_after_seconds)
    target = workers[kill_worker_id]
    target.mark_unhealthy()
    logger.warning(
        "[FailureSim] ⚠  Worker %d marked UNHEALTHY at t+%.1fs",
        kill_worker_id, kill_after_seconds,
    )
    if recover_after_seconds is not None:
        time.sleep(recover_after_seconds)
        target.mark_healthy()
        logger.info(
            "[FailureSim] ✅ Worker %d marked HEALTHY at t+%.1fs",
            kill_worker_id, kill_after_seconds + recover_after_seconds,
        )


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def _percentile(sorted_data: list, pct: float) -> float:
    if not sorted_data:
        return 0.0
    idx = max(0, int(len(sorted_data) * pct / 100) - 1)
    return sorted_data[idx]


# ---------------------------------------------------------------------------
# Main load-test runner
# ---------------------------------------------------------------------------

def run_load_test(
    scheduler,
    num_users: int = 1000,
    stagger_delay: float = 0.15,
    max_concurrent: Optional[int] = None,
    # failure simulation
    workers: Optional[list] = None,
    kill_worker_id: Optional[int] = None,
    kill_after_seconds: float = 5.0,
    recover_after_seconds: Optional[float] = 15.0,
) -> dict:
    """Launch *num_users* concurrent threads and benchmark the system.

    Returns
    -------
    dict
        Keys: num_users, elapsed, success_count, fail_count, throughput,
              avg_latency, p50_latency, p95_latency, p99_latency,
              worker_counts (dict[worker_id -> int]).
    """
    print(f"\n{'=' * 62}")
    print(f"  LOAD TEST STARTING  |  Concurrent users: {num_users}")
    if stagger_delay > 0.0:
        print(f"  Stagger delay: {stagger_delay}s  (~{1/stagger_delay:.0f} req/s launch rate)")
    if max_concurrent:
        print(f"  Max in-flight   : {max_concurrent}")
    if kill_worker_id is not None:
        print(f"  Failure sim     : Worker {kill_worker_id} killed at t+{kill_after_seconds}s"
              + (f", recovered at t+{kill_after_seconds + recover_after_seconds:.0f}s"
                 if recover_after_seconds else " (no recovery)"))
    print(f"{'=' * 62}\n")

    results: List = []
    lock = threading.Lock()
    semaphore = threading.Semaphore(max_concurrent) if max_concurrent else None

    # -- Optional failure simulation thread ------------------------------
    if kill_worker_id is not None:
        if workers is None:
            raise ValueError("Pass 'workers' list when using kill_worker_id.")
        sim_thread = threading.Thread(
            target=_failure_simulation_thread,
            args=(workers, kill_worker_id, kill_after_seconds, recover_after_seconds),
            daemon=True,
        )
        sim_thread.start()

    # -- Launch user threads ---------------------------------------------
    start_time = time.time()
    threads: List[threading.Thread] = []

    for i in range(num_users):
        t = threading.Thread(
            target=simulate_user,
            args=(scheduler, i, results, lock, semaphore),
        )
        threads.append(t)
        t.start()
        if stagger_delay > 0.0 and i < num_users - 1:
            time.sleep(stagger_delay)

    for t in threads:
        t.join()

    elapsed = time.time() - start_time

    # -- Compute metrics -------------------------------------------------
    success_responses = [r for _, r in results if r.success]
    fail_count        = sum(1 for _, r in results if not r.success)
    success_count     = len(success_responses)
    latencies         = sorted(r.latency for r in success_responses)

    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p50_latency = _percentile(latencies, 50)
    p95_latency = _percentile(latencies, 95)
    p99_latency = _percentile(latencies, 99)
    throughput  = success_count / elapsed if elapsed > 0 else 0.0

    # -- Per-worker distribution -----------------------------------------
    worker_counts: Dict[int, int] = defaultdict(int)
    for r in success_responses:
        wid = getattr(r, "worker_id", None)
        if wid is not None:
            worker_counts[wid] += 1

    # -- Print summary ---------------------------------------------------
    print(f"\n{'=' * 62}")
    print(f"  LOAD TEST COMPLETE")
    print(f"{'=' * 62}")
    print(f"    Users           : {num_users}")
    print(f"    Total time      : {elapsed:.2f}s")
    print(f"    Successful      : {success_count}")
    print(f"    Failed          : {fail_count}")
    print(f"    Throughput      : {throughput:.2f} req/s")
    print(f"    Avg latency     : {avg_latency:.4f}s")
    print(f"    P50 latency     : {p50_latency:.4f}s")
    print(f"    P95 latency     : {p95_latency:.4f}s")
    print(f"    P99 latency     : {p99_latency:.4f}s")
    if worker_counts:
        print(f"    Worker dist     : { {k: worker_counts[k] for k in sorted(worker_counts)} }")
    print(f"{'=' * 62}\n")

    return {
        "num_users":     num_users,
        "elapsed":       elapsed,
        "success_count": success_count,
        "fail_count":    fail_count,
        "throughput":    throughput,
        "avg_latency":   avg_latency,
        "p50_latency":   p50_latency,
        "p95_latency":   p95_latency,
        "p99_latency":   p99_latency,
        "worker_counts": dict(worker_counts),
    }