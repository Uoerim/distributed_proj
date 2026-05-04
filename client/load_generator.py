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

    results: List = []
    lock = threading.Lock()
    threads: List[threading.Thread] = []

    start_time = time.time()

    for i in range(num_users):
        t = threading.Thread(
            target=simulate_user,
            args=(scheduler, i, results, lock),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start_time

    # -- Summary ----------------------------------------------------------
    success_count = sum(1 for _, r in results if r.success)
    fail_count = sum(1 for _, r in results if not r.success)
    latencies = [r.latency for _, r in results if r.success]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print(f"\n{'=' * 62}")
    print(f"  LOAD TEST COMPLETE")
    print(f"{'=' * 62}")
    print(f"    Users           : {num_users}")
    print(f"    Total time      : {elapsed:.2f}s")
    print(f"    Successful      : {success_count}")
    print(f"    Failed          : {fail_count}")
    print(f"    Avg latency     : {avg_latency:.4f}s")
    print(f"    Throughput      : {success_count / elapsed:.2f} req/s")
    print(f"{'=' * 62}\n")
