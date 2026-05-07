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
* Per-request latency and success/failure logging.
* Aggregate summary with P50 / P95 / P99 latency percentiles printed
  after the test completes.
* Optional failure simulation: kills a chosen worker mid-test, then
  optionally recovers it to verify fault-tolerance & task reassignment.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from typing import List, Optional

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
    """Simulate a single user sending one request.

    Parameters
    ----------
    scheduler : Scheduler
        The master scheduler instance.
    user_id : int
        Unique identifier for this simulated user.
    results : list
        Shared list to collect (user_id, response) tuples.
    lock : threading.Lock
        Guard for the shared *results* list.
    semaphore : threading.Semaphore, optional
        When provided, limits the number of requests that are
        *in-flight* at the same time (max-concurrency cap).
    """
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
                    user_id,
                    response.id,
                    response.latency,
                )
            else:
                logger.warning(
                    "[Client] User %d | FAILED | %s",
                    user_id,
                    response.result[:60],
                )

        except Exception as exc:
            logger.error("[Client] User %d | Exception: %s", user_id, exc)


class _NullContext:
    """No-op context manager used when no semaphore is needed."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Failure simulation helper
# ---------------------------------------------------------------------------

def _failure_simulation_thread(
    workers: list,
    kill_worker_id: int,
    kill_after_seconds: float,
    recover_after_seconds: Optional[float],
) -> None:
    """Background thread that kills (and optionally recovers) a worker.

    Parameters
    ----------
    workers : list[GPUWorker]
        Full list of worker objects.
    kill_worker_id : int
        Index of the worker to kill.
    kill_after_seconds : float
        Seconds after load-test start to trigger the failure.
    recover_after_seconds : float or None
        If set, seconds *after the kill* to call ``mark_healthy()``
        and re-register the worker.  Pass ``None`` to skip recovery.
    """
    time.sleep(kill_after_seconds)

    target = workers[kill_worker_id]
    target.mark_unhealthy()
    logger.warning(
        "[FailureSim] ⚠  Worker %d marked UNHEALTHY at t+%.1fs",
        kill_worker_id,
        kill_after_seconds,
    )

    if recover_after_seconds is not None:
        time.sleep(recover_after_seconds)
        target.mark_healthy()
        logger.info(
            "[FailureSim] ✅ Worker %d marked HEALTHY (recovered) at t+%.1fs",
            kill_worker_id,
            kill_after_seconds + recover_after_seconds,
        )


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def _percentile(sorted_data: list, pct: float) -> float:
    """Return the *pct*-th percentile (0-100) of a pre-sorted list."""
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
    # --- failure simulation ---
    workers: Optional[list] = None,
    kill_worker_id: Optional[int] = None,
    kill_after_seconds: float = 5.0,
    recover_after_seconds: Optional[float] = 15.0,
) -> dict:
    """Launch *num_users* concurrent threads and benchmark the system.

    Parameters
    ----------
    scheduler : Scheduler
        The master scheduler to send requests to.
    num_users : int
        Number of simulated concurrent users.
    stagger_delay : float
        Seconds to wait between launching each thread.  Defaults to
        0.15 s (~7 req/s).  Set to ``0.0`` to launch all at once.
    max_concurrent : int or None
        If set, caps the number of requests *in-flight* simultaneously
        via a ``threading.Semaphore``.  Useful for very large
        ``num_users`` values to avoid OS thread exhaustion.
    workers : list[GPUWorker] or None
        Full worker list — required when ``kill_worker_id`` is set.
    kill_worker_id : int or None
        Index of the worker to kill mid-test.  Pass ``None`` to skip
        failure simulation entirely.
    kill_after_seconds : float
        Seconds after test start to trigger the worker failure.
    recover_after_seconds : float or None
        Seconds *after the kill* to recover the worker.
        ``None`` means no recovery (worker stays dead for the test).

    Returns
    -------
    dict
        Summary metrics: throughput, avg/p50/p95/p99 latency, counts.
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

    # -- Optional: start failure-simulation thread -----------------------
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
    success_count = sum(1 for _, r in results if r.success)
    fail_count    = sum(1 for _, r in results if not r.success)
    latencies     = sorted(r.latency for _, r in results if r.success)

    avg_latency  = statistics.mean(latencies)  if latencies else 0.0
    p50_latency  = _percentile(latencies, 50)
    p95_latency  = _percentile(latencies, 95)
    p99_latency  = _percentile(latencies, 99)
    throughput   = success_count / elapsed if elapsed > 0 else 0.0

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
    }