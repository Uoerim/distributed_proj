"""
main.py
=======
Main entry point for the Distributed LLM Load Balancing System.

Wires together all system components and runs the load test.

Usage
-----
    python main.py

Optional environment variables
------------------------------
    NUM_WORKERS   Number of GPU worker nodes (default: 4)
    NUM_USERS     Number of simulated concurrent users (default: 100)
"""

from __future__ import annotations

import logging
import os
import sys

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Bootstrap the system and run the load test."""

    num_workers = int(os.environ.get("NUM_WORKERS", 4))
    num_users = int(os.environ.get("NUM_USERS", 100))

    logger.info("=" * 62)
    logger.info("  Distributed LLM Load Balancing System")
    logger.info("  Workers: %d  |  Users: %d", num_workers, num_users)
    logger.info("=" * 62)

    # -- 1. Create GPU workers -------------------------------------------
    from workers.gpu_worker import GPUWorker

    workers = [GPUWorker(i) for i in range(num_workers)]

    # -- 2. Create Load Balancer -----------------------------------------
    from lb.load_balancer import LoadBalancer

    lb = LoadBalancer(workers)

    # -- 3. Create Master Scheduler --------------------------------------
    from master.scheduler import Scheduler

    scheduler = Scheduler(lb)

    # -- 4. Run load test ------------------------------------------------
    from client.load_generator import run_load_test

    run_load_test(scheduler, num_users=num_users)

    # -- 5. Print detailed report ----------------------------------------
    scheduler.print_report()


if __name__ == "__main__":
    main()
