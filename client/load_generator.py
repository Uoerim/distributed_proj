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
* Per-request latency and success/failure logging.
* Aggregate summary printed after the test completes.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List

from common.models import Request

logger = logging.getLogger(__name__)


def simulate_user(scheduler, user_id: int, results: list, lock: threading.Lock) -> None:
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
    """
    request = Request(id=user_id, query=f"Query {user_id}")

    start_time = time.time()

    try:
        response = scheduler.handle_request(request)
        latency = time.time() - start_time

        with lock:
            results.append((user_id, response, latency))

        if response.success:
            logger.debug(
                "[Client] User %d | Worker %s | Latency: %.3fs",
                user_id,
                response.worker_id,
                latency,
            )
        else:
            logger.warning(
                "[Client] User %d | FAILED | %s",
                user_id,
                response.result[:60],
            )

    except Exception as exc:
        latency = time.time() - start_time

        with lock:
            results.append((user_id, None, latency))

        logger.error("[Client] User %d | Exception: %s", user_id, exc)


def run_load_test(scheduler, num_users: int = 1000) -> None:
    """Launch *num_users* concurrent threads and benchmark the system.

    Parameters
    ----------
    scheduler : Scheduler
        The master scheduler to send requests to.
    num_users : int
        Number of simulated concurrent users.
    """
    print(f"\n{'=' * 62}")
    print(f"  LOAD TEST STARTING  |  Concurrent users: {num_users}")
    print(f"{'=' * 62}\n")

    results = []
    lock = threading.Lock()
    threads = []

    start_time = time.time()

    for i in range(1, num_users + 1):
        t = threading.Thread(
            target=simulate_user,
            args=(scheduler, i, results, lock),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    total_time  = time.time() - start_time

    successful = [(u, r, l) for (u, r, l) in results if r and r.success]
    failed = [(u, r, l) for (u, r, l) in results if not r or not r.success]

    latencies = [l for (_, _, l) in successful]

    success_count = len(successful)
    fail_count = len(failed)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0

    throughput = success_count / total_time if total_time > 0 else 0.0

    # ---------- Worker Distribution ----------
    worker_stats = {}
    for (_, response, _) in successful:
        wid = response.worker_id
        worker_stats[wid] = worker_stats.get(wid, 0) + 1

    # ---------- Print Results ----------
    print(f"\n{'=' * 62}")
    print("  LOAD TEST COMPLETE")
    print(f"{'=' * 62}")
    print(f"    Users           : {num_users}")
    print(f"    Total time      : {total_time:.2f}s")
    print(f"    Successful      : {success_count}")
    print(f"    Failed          : {fail_count}")
    print(f"    Avg latency     : {avg_latency:.4f}s")
    print(f"    Min latency     : {min_latency:.4f}s")
    print(f"    Max latency     : {max_latency:.4f}s")
    print(f"    Throughput      : {throughput:.2f} req/s")

    print("\n    Worker distribution:")
    if worker_stats:
        for wid, count in sorted(worker_stats.items()):
            print(f"    Worker {wid}: {count} requests")
    else:
        print("    No successful worker responses recorded.")

    print(f"{'=' * 62}\n")

  