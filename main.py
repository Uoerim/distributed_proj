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
    OLLAMA_HOST_0 Ollama URL for Worker 0 (default: http://localhost:11434)
    OLLAMA_HOST_1 Ollama URL for Worker 1 (e.g. ngrok/Kaggle URL)
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_worker_hosts() -> list[str]:
    """Read OLLAMA_HOST_0, OLLAMA_HOST_1, ... from .env and return as list."""
    hosts = []
    i = 0
    while True:
        host = os.environ.get(f"OLLAMA_HOST_{i}")
        if not host:
            break
        hosts.append(host)
        i += 1

    # Fallback: single host if no numbered vars defined
    if not hosts:
        hosts = [os.environ.get("OLLAMA_HOST", "http://localhost:11434")]

    return hosts


def main() -> None:
    """Bootstrap the system and run the load test."""

    num_users = int(os.environ.get("NUM_USERS", 100))

    # -- 1. Create GPU workers -------------------------------------------
    from workers.gpu_worker import GPUWorker

    hosts = _build_worker_hosts()
    workers = [
        GPUWorker(worker_id=i, ollama_host=host)
        for i, host in enumerate(hosts)
    ]

    num_workers = len(workers)

    logger.info("=" * 62)
    logger.info("  Distributed LLM Load Balancing System")
    logger.info("  Workers: %d  |  Users: %d", num_workers, num_users)
    logger.info("=" * 62)

    for w in workers:
        logger.info("  Worker %d → %s", w.id, w.ollama_host)
    logger.info("=" * 62)

    # -- 2. Create Load Balancer -----------------------------------------
    from lb.load_balancer import LoadBalancer

    lb = LoadBalancer(workers=workers)

    # -- 3. Create Master Scheduler --------------------------------------
    from master.scheduler import Scheduler

    scheduler = Scheduler(lb)

    # -- 4. Run load test ------------------------------------------------
    from client.load_generator import run_load_test

    
    for users in [100, 500, 1000]:
        logger.info("\n🔥 Running test with %d users\n", users)
        run_load_test(scheduler, num_users=users)

    # -- 5. Print detailed report ----------------------------------------
    scheduler.print_report()


if __name__ == "__main__":
    main()