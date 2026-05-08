"""
main.py
=======
Main entry point for the Distributed LLM Load Balancing System.

Usage
-----
    # Single strategy run:
    python main.py --strategy round_robin --users 500

    # Run all three strategies back-to-back and compare:
    python main.py --compare-strategies --users 200

    # Failure simulation:
    python main.py --users 300 --kill-worker 1 --kill-after 5

Optional CLI arguments
----------------------
    --strategy            round_robin | least_connections | load_aware
    --users               Number of simulated concurrent users
    --max-concurrent      Max requests in-flight simultaneously (default: 50)
    --kill-worker         Worker index to kill mid-test
    --kill-after          Seconds before the kill fires (default: 5)
    --no-recovery         Killed worker is NOT recovered during the test
    --compare-strategies  Run all 3 strategies sequentially and print comparison

Optional environment variables
-------------------------------
    NUM_USERS       Number of simulated concurrent users (default: 100)
    OLLAMA_HOST_0   Ollama URL for Worker 0
    OLLAMA_HOST_1   Ollama URL for Worker 1  (etc.)
"""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
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
    hosts = []
    i = 0
    while True:
        host = os.environ.get(f"OLLAMA_HOST_{i}")
        if not host:
            break
        hosts.append(host)
        i += 1
    if not hosts:
        hosts = [os.environ.get("OLLAMA_HOST", "http://localhost:11434")]
    return hosts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed LLM Load Balancing System")
    parser.add_argument(
        "--strategy",
        default=os.environ.get("LB_STRATEGY", "round_robin"),
        choices=["round_robin", "least_connections", "load_aware"],
        help="Load-balancing strategy (default: round_robin)",
    )
    parser.add_argument(
        "--users", type=int,
        default=int(os.environ.get("NUM_USERS", 50)),
        help="Number of simulated concurrent users (default: 50)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=50, dest="max_concurrent",
        help="Max requests in-flight at once (default: 50)",
    )
    parser.add_argument(
        "--kill-worker", type=int, default=None, dest="kill_worker",
        help="Worker index to kill mid-test",
    )
    parser.add_argument(
        "--kill-after", type=float, default=5.0, dest="kill_after",
        help="Seconds after test start to kill the worker (default: 5)",
    )
    parser.add_argument(
        "--no-recovery", action="store_true", dest="no_recovery",
        help="Killed worker is NOT recovered during the test",
    )
    parser.add_argument(
        "--compare-strategies", action="store_true", dest="compare_strategies",
        help="Run all 3 strategies sequentially and print a comparison table",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Single run helper
# ---------------------------------------------------------------------------

def _single_run(scheduler, lb, workers, strategy_name: str, args) -> dict:
    """Switch strategy, run load test, return metrics dict."""
    from common.models import RoutingStrategy
    from client.load_generator import run_load_test

    strategy_map = {
        "round_robin":       RoutingStrategy.ROUND_ROBIN,
        "least_connections": RoutingStrategy.LEAST_CONNECTIONS,
        "load_aware":        RoutingStrategy.LOAD_AWARE,
    }

    lb.set_strategy(strategy_map[strategy_name])
    logger.info("Strategy switched to: %s", strategy_name)

    recover_after = None if args.no_recovery else 15.0

    metrics = run_load_test(
        scheduler,
        num_users       = args.users,
        max_concurrent  = args.max_concurrent,
        workers         = workers if args.kill_worker is not None else None,
        kill_worker_id  = args.kill_worker,
        kill_after_seconds    = args.kill_after,
        recover_after_seconds = recover_after,
    )

    logger.info(
        "METRICS | strategy=%s users=%d throughput=%.2f req/s "
        "p50=%.3fs p95=%.3fs p99=%.3fs failed=%d",
        strategy_name,
        metrics["num_users"],
        metrics["throughput"],
        metrics["p50_latency"],
        metrics["p95_latency"],
        metrics["p99_latency"],
        metrics["fail_count"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # -- 1. Workers ----------------------------------------------------------
    from workers.gpu_worker import GPUWorker

    hosts = _build_worker_hosts()
    workers = [GPUWorker(worker_id=i, ollama_host=h) for i, h in enumerate(hosts)]

    logger.info("=" * 62)
    logger.info("  Distributed LLM Load Balancing System")
    logger.info("  Workers: %d  |  Users: %d", len(workers), args.users)
    logger.info("=" * 62)
    for w in workers:
        logger.info("  Worker %d → %s", w.id, w.ollama_host)
    logger.info("=" * 62)

    # -- 2. Load Balancer ----------------------------------------------------
    from lb.load_balancer import LoadBalancer
    from common.models import RoutingStrategy

    strategy_map = {
        "round_robin":       RoutingStrategy.ROUND_ROBIN,
        "least_connections": RoutingStrategy.LEAST_CONNECTIONS,
        "load_aware":        RoutingStrategy.LOAD_AWARE,
    }
    lb = LoadBalancer(workers=workers, strategy=strategy_map[args.strategy])

    # -- 3. Scheduler --------------------------------------------------------
    from master.scheduler import Scheduler

    scheduler = Scheduler(lb)
    scheduler.start_health_monitor(check_interval=2.0, recovery_retry_interval=10.0)

    # -- 4. Metrics collector ------------------------------------------------
    from client.metrics import MetricsCollector

    collector = MetricsCollector()

    # -- 5. Run test(s) ------------------------------------------------------
    if args.compare_strategies:
        # Run all 3 strategies back-to-back for comparison
        for strat in ["round_robin", "least_connections", "load_aware"]:
            metrics = _single_run(scheduler, lb, workers, strat, args)
            collector.record(strat, metrics)
    else:
        # Single strategy run
        metrics = _single_run(scheduler, lb, workers, args.strategy, args)
        collector.record(args.strategy, metrics)

    # -- 6. Stop health monitor ----------------------------------------------
    scheduler.stop_health_monitor()

    # -- 7. Reports ----------------------------------------------------------
    scheduler.print_report()
    collector.print_comparison()
    collector.print_worker_distribution()

    if args.compare_strategies:
        best = collector.best_strategy()
        print(f"  ✅ Best strategy by throughput: {best}\n")


if __name__ == "__main__":
    main()