"""
main.py
=======
Main entry point for the Distributed LLM Load Balancing System.

Wires together all system components and runs the load test.

Usage
-----
    python main.py [--strategy STRATEGY] [--users N] [--kill-worker ID]

Optional CLI arguments
----------------------
    --strategy      Load-balancing strategy: round_robin | least_connections
                    | load_aware  (default: round_robin)
    --users         Number of simulated concurrent users (default: env
                    var NUM_USERS, fallback 100)
    --max-concurrent  Max requests in-flight simultaneously (default: 50)
    --kill-worker   Worker index to kill mid-test for failure simulation
                    (omit to skip failure simulation)
    --kill-after    Seconds after test start to kill the worker (default: 5)
    --no-recovery   If set, the killed worker is NOT recovered during the test

Optional environment variables
-------------------------------
    NUM_WORKERS   Number of GPU worker nodes (default: 4)
    NUM_USERS     Number of simulated concurrent users (default: 100)
    OLLAMA_HOST_0 Ollama URL for Worker 0 (default: http://localhost:11434)
    OLLAMA_HOST_1 Ollama URL for Worker 1 (e.g. ngrok/Kaggle URL)
"""

from __future__ import annotations

import argparse
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Distributed LLM Load Balancing System",
    )
    parser.add_argument(
        "--strategy",
        default=os.environ.get("LB_STRATEGY", "round_robin"),
        choices=["round_robin", "least_connections", "load_aware"],
        help="Load-balancing strategy (default: round_robin)",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=int(os.environ.get("NUM_USERS", 100)),
        help="Number of simulated concurrent users (default: 100)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        dest="max_concurrent",
        help="Max requests in-flight at once — semaphore cap (default: 50)",
    )
    # Failure simulation
    parser.add_argument(
        "--kill-worker",
        type=int,
        default=None,
        dest="kill_worker",
        help="Worker index to kill mid-test (omit to skip failure sim)",
    )
    parser.add_argument(
        "--kill-after",
        type=float,
        default=5.0,
        dest="kill_after",
        help="Seconds after test start to kill the worker (default: 5)",
    )
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        dest="no_recovery",
        help="If set, the killed worker is NOT recovered during the test",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Bootstrap the system and run the load test."""
    args = _parse_args()

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
    logger.info("  Workers: %d  |  Users: %d", num_workers, args.users)
    logger.info("=" * 62)
    for w in workers:
        logger.info("  Worker %d → %s", w.id, w.ollama_host)
    logger.info("=" * 62)

    # -- 2. Create Load Balancer -----------------------------------------
    from lb.load_balancer import LoadBalancer
    from common.models import RoutingStrategy

    strategy_map = {
        "round_robin":       RoutingStrategy.ROUND_ROBIN,
        "least_connections": RoutingStrategy.LEAST_CONNECTIONS,
        "load_aware":        RoutingStrategy.LOAD_AWARE,
    }
    chosen_strategy = strategy_map[args.strategy]
    lb = LoadBalancer(workers=workers, strategy=chosen_strategy)
    logger.info("LoadBalancer initialised  |  workers=%d  strategy=%s", num_workers, args.strategy)

    # -- 3. Create Master Scheduler --------------------------------------
    from master.scheduler import Scheduler

    scheduler = Scheduler(lb)

    # -- 3a. Start Health Monitor (Phase 3 – Fault Tolerance) -----------
    scheduler.start_health_monitor(check_interval=2.0, recovery_retry_interval=10.0)

    # -- 4. Run load test ------------------------------------------------
    from client.load_generator import run_load_test

    recover_after = None if args.no_recovery else 15.0

    metrics = run_load_test(
        scheduler,
        num_users=args.users,
        max_concurrent=args.max_concurrent,
        # failure simulation
        workers=workers if args.kill_worker is not None else None,
        kill_worker_id=args.kill_worker,
        kill_after_seconds=args.kill_after,
        recover_after_seconds=recover_after,
    )

    # -- 4a. Stop Health Monitor -----------------------------------------
    scheduler.stop_health_monitor()

    # -- 5. Print detailed report ----------------------------------------
    scheduler.print_report()

    # -- 6. Echo final metrics for report/graphs -------------------------
    logger.info("METRICS | strategy=%s users=%d throughput=%.2f req/s "
                "p50=%.3fs p95=%.3fs p99=%.3fs failed=%d",
                args.strategy,
                metrics["num_users"],
                metrics["throughput"],
                metrics["p50_latency"],
                metrics["p95_latency"],
                metrics["p99_latency"],
                metrics["fail_count"])


if __name__ == "__main__":
    main()