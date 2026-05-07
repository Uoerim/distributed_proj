"""
lb/load_balancer.py
===================
Production-grade Load Balancer for the Distributed LLM Inference System.

This module is responsible for distributing incoming ``Request`` objects
across a pool of registered GPU worker nodes.  It implements
**Round Robin**, **Least Connections**, and **Load-Aware** scheduling
strategies, and includes a request staggering mechanism to avoid
overwhelming ngrok tunnels with burst traffic.

Architecture
------------
::

    ┌────────────┐        ┌──────────────────┐        ┌────────────┐
    │   Client   │───────▶│  LoadBalancer     │───────▶│  Worker 0  │
    │  Requests  │        │  (strategy-based) │        ├────────────┤
    └────────────┘        └──────────────────┘        │  Worker 1  │
                                                       ├────────────┤
                                                       │  Worker N  │
                                                       └────────────┘

Design Decisions
----------------
1. **Strategy Pattern** — ``RoutingStrategy`` enum selects the algorithm.
   A single ``_select_worker()`` method delegates to the appropriate
   private strategy method, making it trivial to add new algorithms.

2. **Thread Safety** — A ``threading.Lock`` guards all shared mutable
   state (index counter, connection counters, statistics) so the balancer
   is safe to call from the concurrent load-test harness.

3. **Statistics Tracking** — Every dispatch records success/failure
   counts, total latency, and per-worker hit counts.  This data feeds
   into the monitoring / admin interface planned for Phase 4.

4. **Worker Health** — ``register_worker`` / ``deregister_worker``
   allow dynamic pool changes at runtime; the fault-tolerance module
   (Phase 3) will call ``deregister_worker`` when a node is detected
   as unhealthy.

5. **Request Staggering** — ``dispatch_staggered`` introduces a small
   configurable delay between consecutive dispatches to avoid exceeding
   ngrok free-tier rate limits (~20 req/min per tunnel) when running
   large load tests across remote workers.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from common.models import Request, Response, RoutingStrategy

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker Protocol – structural typing for worker nodes
# ---------------------------------------------------------------------------


@runtime_checkable
class WorkerNode(Protocol):
    """Minimal interface that any GPU worker must satisfy.

    Using a ``Protocol`` (structural sub-typing) decouples the load
    balancer from the concrete ``GPUWorker`` implementation so that
    mock / stub workers can be injected during unit-testing.
    """

    id: int

    def process(self, request: Request) -> Response:  # pragma: no cover
        ...

    def get_load(self) -> float:  # pragma: no cover
        """Return normalised load metric (0.0 = idle, 1.0 = saturated)."""
        ...


# ---------------------------------------------------------------------------
# Load-Balancer Statistics
# ---------------------------------------------------------------------------


@dataclass
class LoadBalancerStats:
    """Aggregated statistics for observability and reporting.

    All counters are updated atomically under the parent
    ``LoadBalancer._lock``.
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    worker_dispatch_count: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    worker_success_count: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    worker_failure_count: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # -- Derived metrics -------------------------------------------------

    @property
    def average_latency(self) -> float:
        """Mean response latency across all completed requests."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency / self.successful_requests

    @property
    def success_rate(self) -> float:
        """Fraction of requests that succeeded (0.0 – 1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of all metrics."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.success_rate, 4),
            "average_latency_s": round(self.average_latency, 6),
            "per_worker_dispatches": dict(self.worker_dispatch_count),
            "per_worker_successes": dict(self.worker_success_count),
            "per_worker_failures": dict(self.worker_failure_count),
        }


# ---------------------------------------------------------------------------
# Connection Tracker (for Least-Connections)
# ---------------------------------------------------------------------------


class _ConnectionTracker:
    """Tracks the number of in-flight (active) connections per worker."""

    def __init__(self) -> None:
        self._counts: Dict[int, int] = defaultdict(int)

    def increment(self, worker_id: int) -> None:
        self._counts[worker_id] += 1

    def decrement(self, worker_id: int) -> None:
        self._counts[worker_id] = max(0, self._counts[worker_id] - 1)

    def get(self, worker_id: int) -> int:
        return self._counts[worker_id]

    def remove(self, worker_id: int) -> None:
        self._counts.pop(worker_id, None)

    def snapshot(self) -> Dict[int, int]:
        return dict(self._counts)


# ---------------------------------------------------------------------------
# Main LoadBalancer Class
# ---------------------------------------------------------------------------


class LoadBalancer:
    """Distributes incoming requests across a pool of GPU worker nodes.

    Parameters
    ----------
    workers : list[WorkerNode]
        Initial pool of available worker nodes.
    strategy : RoutingStrategy
        Algorithm used to select the next worker.  Defaults to
        ``RoutingStrategy.ROUND_ROBIN``.
    stagger_delay : float
        Seconds to wait between consecutive dispatches in
        ``dispatch_staggered``.  Default is 0.1s (10 req/s burst cap).
        Set to 0.0 to disable staggering entirely.

    Examples
    --------
    >>> from workers.gpu_worker import GPUWorker
    >>> pool = [GPUWorker(i) for i in range(4)]
    >>> lb = LoadBalancer(pool)
    >>> lb.dispatch(Request(id=1, query="Hello"))
    Response(...)
    """

    # ------------------------------------------------------------------ #
    #  Construction / Lifecycle
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        workers: List[WorkerNode],
        strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN,
        stagger_delay: float = 0.1,
    ) -> None:
        if not workers:
            raise ValueError("LoadBalancer requires at least one worker node.")

        self._workers: List[WorkerNode] = list(workers)
        self._strategy: RoutingStrategy = strategy
        self._stagger_delay: float = stagger_delay

        # Round-Robin state
        self._rr_index: int = 0

        # Connection tracking (Least-Connections)
        self._connections = _ConnectionTracker()

        # Request tracking for fault tolerance
        self._request_to_worker: Dict[str, int] = {}

        # Thread safety
        self._lock = threading.Lock()

        # Observability
        self._stats = LoadBalancerStats()

        logger.info(
            "LoadBalancer initialised  |  workers=%d  strategy=%s",
            len(self._workers),
            self._strategy.value,
        )

    # ------------------------------------------------------------------ #
    #  Worker Pool Management
    # ------------------------------------------------------------------ #

    def register_worker(self, worker: WorkerNode) -> None:
        """Add a worker to the active pool at runtime."""
        with self._lock:
            if any(w.id == worker.id for w in self._workers):
                logger.warning("Worker %d is already registered – skipping.", worker.id)
                return
            self._workers.append(worker)
            logger.info(
                "Worker %d registered  |  pool_size=%d",
                worker.id,
                len(self._workers),
            )

    def deregister_worker(self, worker_id: int) -> Optional[WorkerNode]:
        """Remove a worker from the active pool.

        Returns the removed ``WorkerNode`` or ``None`` if the id was
        not found.  Called by the fault-tolerance module when a node
        is detected as unhealthy.
        """
        with self._lock:
            for idx, w in enumerate(self._workers):
                if w.id == worker_id:
                    removed = self._workers.pop(idx)
                    self._connections.remove(worker_id)
                    if self._rr_index >= len(self._workers) and self._workers:
                        self._rr_index = 0
                    logger.info(
                        "Worker %d deregistered  |  pool_size=%d",
                        worker_id,
                        len(self._workers),
                    )
                    return removed
            logger.warning("Attempted to deregister unknown worker %d.", worker_id)
            return None

    @property
    def worker_count(self) -> int:
        """Number of workers currently in the pool."""
        with self._lock:
            return len(self._workers)

    @property
    def worker_ids(self) -> List[int]:
        """Sorted list of active worker identifiers."""
        with self._lock:
            return sorted(w.id for w in self._workers)

    def track_request(self, request_uid: str, worker_id: int) -> None:
        with self._lock:
            self._request_to_worker[request_uid] = worker_id

    def untrack_request(self, request_uid: str) -> None:
        with self._lock:
            self._request_to_worker.pop(request_uid, None)

    def get_requests_on_worker(self, worker_id: int) -> List[str]:
        with self._lock:
            return [uid for uid, wid in self._request_to_worker.items() if wid == worker_id]

    # ------------------------------------------------------------------ #
    #  Strategy Management
    # ------------------------------------------------------------------ #

    def set_strategy(self, strategy: RoutingStrategy) -> None:
        """Switch the routing strategy at runtime."""
        with self._lock:
            old = self._strategy
            self._strategy = strategy
            logger.info(
                "Routing strategy changed  |  %s → %s",
                old.value,
                strategy.value,
            )

    @property
    def current_strategy(self) -> RoutingStrategy:
        """Return the currently active routing strategy."""
        return self._strategy

    # ------------------------------------------------------------------ #
    #  Core Dispatch Logic
    # ------------------------------------------------------------------ #

    def dispatch(self, request: Request) -> Response:
        """Select a worker and forward *request* for processing.

        Steps:
        1. Select a worker using the active routing strategy.
        2. Increment in-flight connection counter.
        3. Delegate ``request`` to the selected worker's ``process()``.
        4. Record success/failure and latency in ``_stats``.
        5. Return the ``Response`` to the caller.
        """
        with self._lock:
            if not self._workers:
                raise RuntimeError("No workers available in the pool.")
            worker = self._select_worker()
            worker_id = worker.id
            self._connections.increment(worker_id)
            self._stats.total_requests += 1
            self._stats.worker_dispatch_count[worker_id] += 1

        dispatch_time = time.time()

        logger.debug(
            "[LB] Request %d (uid=%s) → Worker %d  [strategy=%s]",
            request.id,
            request.uid[:8],
            worker_id,
            self._strategy.value,
        )

        try:
            self.track_request(request.uid, worker_id)
            response = worker.process(request)
            self.untrack_request(request.uid)
            elapsed = time.time() - dispatch_time

            with self._lock:
                self._connections.decrement(worker_id)
                self._stats.successful_requests += 1
                self._stats.worker_success_count[worker_id] += 1
                self._stats.total_latency += elapsed

            logger.debug(
                "[LB] Request %d completed by Worker %d in %.4fs",
                request.id,
                worker_id,
                elapsed,
            )
            return response

        except Exception as exc:
            self.untrack_request(request.uid)
            elapsed = time.time() - dispatch_time

            with self._lock:
                self._connections.decrement(worker_id)
                self._stats.failed_requests += 1
                self._stats.worker_failure_count[worker_id] += 1

            logger.error(
                "[LB] Request %d FAILED on Worker %d after %.4fs: %s",
                request.id,
                worker_id,
                elapsed,
                exc,
            )

            return Response(
                id=request.id,
                request_uid=request.uid,
                result=f"ERROR: {exc}",
                latency=elapsed,
                worker_id=worker_id,
                success=False,
            )

    def dispatch_staggered(self, request: Request) -> Response:
        """Dispatch with a pre-sleep delay to avoid ngrok rate limiting.

        Identical to ``dispatch`` but sleeps for ``self._stagger_delay``
        seconds before forwarding the request.  Use this in the load
        generator when running large tests (250+ users) against remote
        ngrok-tunnelled workers to stay under the free-tier rate limit
        of ~20 requests/minute per tunnel.

        Parameters
        ----------
        request : Request
            The incoming request to be processed.

        Returns
        -------
        Response
            Same as ``dispatch``.

        Example
        -------
        In ``client/load_generator.py``, replace::

            response = scheduler.dispatch(request)

        with::

            response = scheduler.dispatch_staggered(request)

        Or set ``stagger_delay=0.0`` to fall back to instant dispatch.
        """
        if self._stagger_delay > 0.0:
            time.sleep(self._stagger_delay)
        return self.dispatch(request)

    # ------------------------------------------------------------------ #
    #  Worker-Selection Strategies (private)
    # ------------------------------------------------------------------ #

    def _select_worker(self) -> WorkerNode:
        """Delegate to the strategy-specific selection method.

        Must be called while ``self._lock`` is held.
        """
        if self._strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_worker_round_robin()
        elif self._strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return self._select_worker_least_connections()
        elif self._strategy == RoutingStrategy.LOAD_AWARE:
            return self._select_worker_load_aware()
        else:
            raise ValueError(f"Unknown routing strategy: {self._strategy}")

    # -- Round Robin (Phase 2 – fully implemented) -----------------------

    def _select_worker_round_robin(self) -> WorkerNode:
        """Classic Round-Robin: cycle through workers in order.

        Time complexity: O(1).
        """
        worker = self._workers[self._rr_index]
        self._rr_index = (self._rr_index + 1) % len(self._workers)
        return worker

    # -- Least Connections (Phase 3 – implemented) ----------------------

    def _select_worker_least_connections(self) -> WorkerNode:
        """Select the worker with the fewest in-flight connections.

        Iterates all workers and picks the one with the minimum active
        connection count.  Ties are broken by worker id for determinism.

        Time complexity: O(n).
        """
        best_worker = self._workers[0]
        best_count = self._connections.get(best_worker.id)

        for w in self._workers[1:]:
            count = self._connections.get(w.id)
            if count < best_count or (count == best_count and w.id < best_worker.id):
                best_worker = w
                best_count = count

        logger.debug(
            "[LB] Least-Connections selected Worker %d  (active=%d)",
            best_worker.id,
            best_count,
        )
        return best_worker

    # -- Load-Aware (Phase 3 – implemented) -----------------------------

    def _select_worker_load_aware(self) -> WorkerNode:
        """Select the worker with the lowest reported system load.

        Calls ``w.get_load()`` on every worker in the pool and returns
        the one with the minimum value.  If all workers report a load
        above 0.95 (saturated), logs a warning and falls back to
        Round Robin to avoid stalling the request.

        Each ``WorkerNode`` must implement ``get_load() -> float``
        returning a normalised metric where 0.0 = idle, 1.0 = saturated.

        Time complexity: O(n).
        """
        SATURATION_THRESHOLD = 0.95

        loads: List[tuple[float, WorkerNode]] = []
        for w in self._workers:
            try:
                load = w.get_load()
            except (AttributeError, NotImplementedError):
                # Worker does not implement get_load() — treat as 0.0
                load = 0.0
            loads.append((load, w))

        min_load, best_worker = min(loads, key=lambda x: (x[0], x[1].id))

        if min_load >= SATURATION_THRESHOLD:
            logger.warning(
                "[LB] All workers saturated (min load=%.2f) — "
                "falling back to Round Robin.",
                min_load,
            )
            return self._select_worker_round_robin()

        logger.debug(
            "[LB] Load-Aware selected Worker %d  (load=%.2f)",
            best_worker.id,
            min_load,
        )
        return best_worker

    # ------------------------------------------------------------------ #
    #  Observability / Reporting
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> LoadBalancerStats:
        """Return the live statistics object (not a copy)."""
        return self._stats

    def get_stats_summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of all metrics."""
        with self._lock:
            summary = self._stats.summary()
            summary["active_connections"] = self._connections.snapshot()
            summary["pool_size"] = len(self._workers)
            summary["strategy"] = self._strategy.value
            return summary

    def reset_stats(self) -> None:
        """Zero-out all counters.  Useful between benchmark runs."""
        with self._lock:
            self._stats = LoadBalancerStats()
            logger.info("LoadBalancer statistics reset.")

    # ------------------------------------------------------------------ #
    #  Dunder helpers
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"LoadBalancer(workers={len(self._workers)}, "
            f"strategy={self._strategy.value}, "
            f"dispatched={self._stats.total_requests})"
        )
