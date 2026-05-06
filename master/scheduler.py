"""
master/scheduler.py
===================
Master Scheduler (Controller Node) for the Distributed LLM Inference System.

The Scheduler sits between the **Client Layer** and the **Load Balancer**,
acting as the central orchestration point for all incoming requests.
Its responsibilities include:

* Accepting requests from the client layer.
* Validating and enriching requests before dispatch.
* Delegating worker selection to the ``LoadBalancer``.
* Tracking per-request lifecycle (pending → dispatched → completed/failed).
* Maintaining aggregate success/failure counters and latency statistics.
* Providing an observability API for the monitoring dashboard.
* Supporting both synchronous and asynchronous (batch) dispatch modes.

Architecture
------------
::

    ┌──────────┐     ┌────────────────┐     ┌──────────────┐     ┌──────────┐
    │  Client  │────▶│   Scheduler    │────▶│ LoadBalancer  │────▶│ Workers  │
    │  Layer   │◀────│  (this module) │◀────│              │◀────│          │
    └──────────┘     └────────────────┘     └──────────────┘     └──────────┘

Design Decisions
----------------
1. **Single Responsibility** — The Scheduler handles *orchestration*
   (validation, lifecycle, stats).  It never touches worker-selection
   logic — that is entirely the Load Balancer's job.

2. **Thread Safety** — All shared state is guarded by a
   ``threading.Lock`` so the scheduler is safe for concurrent use
   from the load-test harness.

3. **Request Registry** — A lightweight in-memory dict maps every
   ``request.uid`` to its ``Response`` (or ``None`` while in-flight).
   This enables the fault-tolerance module (Phase 3) to detect
   orphaned requests and re-dispatch them.

4. **Batch Dispatch** — ``handle_batch()`` processes a list of
   requests sequentially (or could be made parallel later) and
   returns aggregated results, which is useful for throughput
   benchmarks.

5. **Callbacks** — An optional ``on_complete`` callback list lets
   other modules (monitoring, logging, metrics exporters) subscribe
   to dispatch events without tight coupling.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from common.models import Request, RequestStatus, Response, RoutingStrategy
from lb.load_balancer import LoadBalancer

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheduler Statistics
# ---------------------------------------------------------------------------

@dataclass
class SchedulerStats:
    """Aggregate metrics tracked by the Scheduler.

    All fields are updated under the parent ``Scheduler._lock``.
    """

    total_received: int = 0
    total_dispatched: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_latency: float = 0.0

    # Timestamps for uptime / throughput calculation
    first_request_at: Optional[float] = None
    last_request_at: Optional[float] = None

    @property
    def average_latency(self) -> float:
        """Mean end-to-end latency for successfully completed requests."""
        if self.total_completed == 0:
            return 0.0
        return self.total_latency / self.total_completed

    @property
    def success_rate(self) -> float:
        """Fraction of dispatched requests that succeeded (0.0 – 1.0)."""
        dispatched = self.total_completed + self.total_failed
        if dispatched == 0:
            return 0.0
        return self.total_completed / dispatched

    @property
    def throughput(self) -> float:
        """Requests completed per second since the first request."""
        if (
            self.first_request_at is None
            or self.last_request_at is None
            or self.total_completed == 0
        ):
            return 0.0
        elapsed = self.last_request_at - self.first_request_at
        if elapsed <= 0:
            return float(self.total_completed)
        return self.total_completed / elapsed

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of all scheduler metrics."""
        return {
            "total_received": self.total_received,
            "total_dispatched": self.total_dispatched,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "success_rate": round(self.success_rate, 4),
            "average_latency_s": round(self.average_latency, 6),
            "throughput_rps": round(self.throughput, 2),
        }


# ---------------------------------------------------------------------------
# Request Registry Entry
# ---------------------------------------------------------------------------

@dataclass
class _RequestRecord:
    """Internal bookkeeping for a single request's lifecycle."""

    request: Request
    response: Optional[Response] = None
    dispatched_at: Optional[float] = None
    completed_at: Optional[float] = None
    worker_id: Optional[int] = None
    status: RequestStatus = RequestStatus.PENDING
    retry_count: int = 0


# ---------------------------------------------------------------------------
# Main Scheduler Class
# ---------------------------------------------------------------------------

class Scheduler:
    """Central orchestrator that receives requests and coordinates dispatch.

    Parameters
    ----------
    load_balancer : LoadBalancer
        The load-balancer instance responsible for worker selection.
    max_retries : int
        Number of automatic retry attempts on worker failure (0 = no retry).
        Full retry logic is a Phase 3 deliverable; the parameter is
        accepted now so the interface is stable.

    Examples
    --------
    >>> from lb.load_balancer import LoadBalancer
    >>> from workers.gpu_worker import GPUWorker
    >>> workers = [GPUWorker(i) for i in range(4)]
    >>> lb = LoadBalancer(workers)
    >>> scheduler = Scheduler(lb)
    >>> response = scheduler.handle_request(Request(id=1, query="Hello"))
    """

    # ------------------------------------------------------------------ #
    #  Construction / Lifecycle
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        load_balancer: LoadBalancer,
        max_retries: int = 0,
    ) -> None:
        self._lb: LoadBalancer = load_balancer
        self._max_retries: int = max_retries

        # Thread safety
        self._lock = threading.Lock()

        # Observability
        self._stats = SchedulerStats()

        # Request lifecycle tracking  {request_uid: _RequestRecord}
        self._registry: Dict[str, _RequestRecord] = {}

        # Event callbacks – invoked after every dispatch completes
        self._on_complete_callbacks: List[Callable[[Request, Response], None]] = []

        # Health monitoring (Phase 3 - Fault Tolerance)
        self._health_monitor_thread: Optional[threading.Thread] = None
        self._monitor_active: bool = False
        self._failed_workers: Dict[int, float] = {}
        self._worker_pool_cache: Dict[int, Any] = {}
        self._check_interval: float = 2.0
        self._recovery_retry_interval: float = 10.0

        logger.info(
            "Scheduler initialised  |  lb=%r  max_retries=%d",
            self._lb,
            self._max_retries,
        )

    # ------------------------------------------------------------------ #
    #  Health Monitoring (Phase 3 - Fault Tolerance)
    # ------------------------------------------------------------------ #

    def start_health_monitor(self, check_interval: float = 2.0, recovery_retry_interval: float = 10.0) -> None:
        if self._monitor_active:
            logger.warning("Health monitor already running.")
            return
        self._monitor_active = True
        self._check_interval = check_interval
        self._recovery_retry_interval = recovery_retry_interval
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="SchedulerHealthMonitor"
        )
        self._health_monitor_thread.start()
        logger.info("Health monitor started (interval=%.1fs)", check_interval)

    def stop_health_monitor(self) -> None:
        if not self._monitor_active:
            return
        self._monitor_active = False
        if self._health_monitor_thread:
            self._health_monitor_thread.join(timeout=5.0)
        logger.info("Health monitor stopped.")

    def _health_monitor_loop(self) -> None:
        while self._monitor_active:
            try:
                self._cache_worker_pool()
                self._check_worker_health()
                self._attempt_worker_recovery()
                self._retry_orphaned_requests()
            except Exception as exc:
                logger.error("Error in health monitor loop: %s", exc)
            time.sleep(self._check_interval)

    def _cache_worker_pool(self) -> None:
        with self._lock:
            for worker_id in self._lb.worker_ids:
                if worker_id not in self._worker_pool_cache:
                    for w in self._lb._workers:
                        if w.id == worker_id:
                            self._worker_pool_cache[worker_id] = w
                            break

    def _check_worker_health(self) -> None:
        with self._lock:
            active_worker_ids = list(self._lb.worker_ids)
        
        for worker_id in active_worker_ids:
            if worker_id in self._worker_pool_cache:
                worker = self._worker_pool_cache[worker_id]
                try:
                    if not worker.is_healthy():
                        self._fail_worker(worker_id)
                except Exception as exc:
                    logger.warning("Health check failed for worker %d: %s", worker_id, exc)
                    self._fail_worker(worker_id)

    def _attempt_worker_recovery(self) -> None:
        now = time.time()
        failed_ids = list(self._failed_workers.keys())
        
        for worker_id in failed_ids:
            time_failed = self._failed_workers[worker_id]
            if now - time_failed >= self._recovery_retry_interval:
                if worker_id in self._worker_pool_cache:
                    worker = self._worker_pool_cache[worker_id]
                    try:
                        if worker.is_healthy():
                            self._recover_worker(worker_id)
                    except Exception as exc:
                        logger.debug("Recovery check failed for worker %d: %s", worker_id, exc)

    def _fail_worker(self, worker_id: int) -> None:
        if worker_id in self._failed_workers:
            return
        with self._lock:
            self._lb.deregister_worker(worker_id)
        self._failed_workers[worker_id] = time.time()
        logger.warning("[Health] Worker %d failed and deregistered. Remaining: %d", 
                      worker_id, self._lb.worker_count)

    def _recover_worker(self, worker_id: int) -> None:
        if worker_id not in self._failed_workers:
            return
        if worker_id in self._worker_pool_cache:
            worker = self._worker_pool_cache[worker_id]
            with self._lock:
                self._lb.register_worker(worker)
            del self._failed_workers[worker_id]
            logger.info("[Health] Worker %d recovered and re-registered. Active: %d", 
                       worker_id, self._lb.worker_count)

    def _retry_orphaned_requests(self) -> None:
        failed_worker_ids = list(self._failed_workers.keys())
        
        for worker_id in failed_worker_ids:
            orphaned_uids = self._lb.get_requests_on_worker(worker_id)
            
            for uid in orphaned_uids:
                with self._lock:
                    record = self._registry.get(uid)
                    if not record:
                        continue
                    
                    if record.retry_count >= 3:
                        logger.warning("[Fault] Request %d max retries exceeded (3)", record.request.id)
                        continue
                    
                    record.retry_count += 1
                
                logger.info("[Fault] Retrying orphaned request %d (retry %d/3)", 
                           record.request.id, record.retry_count)
                
                self._lb.untrack_request(uid)
                response = self._lb.dispatch(record.request)
                
                with self._lock:
                    record.response = response
                    record.completed_at = time.time()
                    record.worker_id = response.worker_id
                    
                    if response.success:
                        record.status = RequestStatus.COMPLETED
                        record.request.status = RequestStatus.COMPLETED
                        self._stats.total_completed += 1
                        self._stats.total_latency += response.latency
                    else:
                        record.status = RequestStatus.FAILED
                        record.request.status = RequestStatus.FAILED
                        self._stats.total_failed += 1

    def get_failed_workers(self) -> Dict[int, float]:
        return dict(self._failed_workers)

    # ------------------------------------------------------------------ #
    #  Callback Registration
    # ------------------------------------------------------------------ #

    def register_callback(
        self, callback: Callable[[Request, Response], None]
    ) -> None:
        """Register a function to be called after each dispatch completes.

        Parameters
        ----------
        callback : Callable[[Request, Response], None]
            A function that receives the original ``Request`` and the
            resulting ``Response``.  Useful for logging, monitoring,
            or triggering downstream pipelines.
        """
        self._on_complete_callbacks.append(callback)
        logger.debug("Callback registered: %s", callback.__name__)

    # ------------------------------------------------------------------ #
    #  Single Request Handling
    # ------------------------------------------------------------------ #

    def handle_request(self, request: Request) -> Response:
        """Accept, validate, dispatch, and track a single request.

        This is the primary entry-point invoked by the Client Layer.

        Workflow
        --------
        1. Validate the request.
        2. Register it in the in-memory registry.
        3. Update statistics (received counter, timestamps).
        4. Delegate to ``LoadBalancer.dispatch()``.
        5. Record the outcome (success / failure, latency).
        6. Fire registered callbacks.
        7. Return the ``Response`` to the caller.

        Parameters
        ----------
        request : Request
            The incoming client request.

        Returns
        -------
        Response
            The processed response from the selected worker.
        """
        now = time.time()

        # -- Validate -----------------------------------------------------
        self._validate_request(request)

        # -- Register & update pre-dispatch stats -------------------------
        with self._lock:
            record = _RequestRecord(request=request, dispatched_at=now)
            self._registry[request.uid] = record

            self._stats.total_received += 1
            if self._stats.first_request_at is None:
                self._stats.first_request_at = now

        logger.info(
            "[Scheduler] Dispatching request %d  (uid=%s, query='%s')",
            request.id,
            request.uid[:8],
            request.query[:50],
        )

        # -- Dispatch via Load Balancer -----------------------------------
        request.status = RequestStatus.DISPATCHED

        with self._lock:
            self._stats.total_dispatched += 1

        response = self._lb.dispatch(request)

        # -- Record outcome -----------------------------------------------
        completed_at = time.time()

        with self._lock:
            record.response = response
            record.completed_at = completed_at
            record.worker_id = response.worker_id
            self._stats.last_request_at = completed_at

            if response.success:
                record.status = RequestStatus.COMPLETED
                request.status = RequestStatus.COMPLETED
                self._stats.total_completed += 1
                self._stats.total_latency += response.latency
            else:
                record.status = RequestStatus.FAILED
                request.status = RequestStatus.FAILED
                self._stats.total_failed += 1

        # -- Log result ---------------------------------------------------
        if response.success:
            logger.info(
                "[Scheduler] Request %d completed  |  worker=%d  latency=%.4fs",
                request.id,
                response.worker_id,
                response.latency,
            )
        else:
            logger.warning(
                "[Scheduler] Request %d FAILED  |  worker=%d  reason=%s",
                request.id,
                response.worker_id,
                response.result[:80],
            )

        # -- Fire callbacks -----------------------------------------------
        self._fire_callbacks(request, response)

        return response

    # ------------------------------------------------------------------ #
    #  Batch Request Handling
    # ------------------------------------------------------------------ #

    def handle_batch(self, requests: List[Request]) -> List[Response]:
        """Dispatch a batch of requests sequentially and return all responses.

        Parameters
        ----------
        requests : list[Request]
            Ordered list of requests to dispatch.

        Returns
        -------
        list[Response]
            Responses in the same order as the input requests.
        """
        logger.info(
            "[Scheduler] Batch dispatch started  |  batch_size=%d",
            len(requests),
        )
        responses: List[Response] = []
        for req in requests:
            responses.append(self.handle_request(req))
        logger.info(
            "[Scheduler] Batch dispatch complete  |  success=%d  failed=%d",
            sum(1 for r in responses if r.success),
            sum(1 for r in responses if not r.success),
        )
        return responses

    # ------------------------------------------------------------------ #
    #  Request Lifecycle Queries
    # ------------------------------------------------------------------ #

    def get_request_status(self, uid: str) -> Optional[RequestStatus]:
        """Look up the current status of a request by its UID.

        Returns ``None`` if the UID is unknown.
        """
        with self._lock:
            record = self._registry.get(uid)
            return record.status if record else None

    def get_request_record(self, uid: str) -> Optional[Dict[str, Any]]:
        """Return a read-only snapshot of a request's lifecycle data.

        Useful for debugging and the monitoring dashboard.
        """
        with self._lock:
            record = self._registry.get(uid)
            if record is None:
                return None
            return {
                "request_id": record.request.id,
                "uid": record.request.uid,
                "query": record.request.query,
                "status": record.status.name,
                "dispatched_at": record.dispatched_at,
                "completed_at": record.completed_at,
                "worker_id": record.worker_id,
                "latency": record.response.latency if record.response else None,
                "success": record.response.success if record.response else None,
            }

    def get_pending_count(self) -> int:
        """Number of requests currently in-flight (dispatched but not done)."""
        with self._lock:
            return sum(
                1
                for r in self._registry.values()
                if r.status in (RequestStatus.PENDING, RequestStatus.DISPATCHED)
            )

    # ------------------------------------------------------------------ #
    #  Strategy Delegation
    # ------------------------------------------------------------------ #

    def set_routing_strategy(self, strategy: RoutingStrategy) -> None:
        """Change the load-balancer's routing strategy at runtime.

        This is a convenience pass-through so that external controllers
        (e.g. an admin CLI or auto-tuner) can switch strategies without
        needing a direct reference to the ``LoadBalancer``.
        """
        self._lb.set_strategy(strategy)
        logger.info(
            "[Scheduler] Routing strategy set to %s",
            strategy.value,
        )

    # ------------------------------------------------------------------ #
    #  Worker Pool Delegation
    # ------------------------------------------------------------------ #

    def register_worker(self, worker: Any) -> None:
        """Pass-through to ``LoadBalancer.register_worker``."""
        self._lb.register_worker(worker)

    def deregister_worker(self, worker_id: int) -> Any:
        """Pass-through to ``LoadBalancer.deregister_worker``."""
        return self._lb.deregister_worker(worker_id)

    # ------------------------------------------------------------------ #
    #  Observability / Reporting
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> SchedulerStats:
        """Return the live statistics object."""
        return self._stats

    def get_stats_summary(self) -> Dict[str, Any]:
        """Return a combined snapshot of Scheduler + LoadBalancer metrics.

        Thread-safe: reads are performed under the lock.
        """
        with self._lock:
            scheduler_metrics = self._stats.summary()

        lb_metrics = self._lb.get_stats_summary()

        return {
            "scheduler": scheduler_metrics,
            "load_balancer": lb_metrics,
            "registry_size": len(self._registry),
        }

    def reset_stats(self) -> None:
        """Reset all counters on both the Scheduler and LoadBalancer."""
        with self._lock:
            self._stats = SchedulerStats()
            self._registry.clear()
        self._lb.reset_stats()
        logger.info("Scheduler and LoadBalancer statistics reset.")

    def print_report(self) -> None:
        """Print a human-readable performance report to stdout.

        Useful for quick inspection after a benchmark run.
        """
        summary = self.get_stats_summary()
        sched = summary["scheduler"]
        lb = summary["load_balancer"]

        border = "=" * 62
        print(f"\n{border}")
        print("            DISTRIBUTED LLM SYSTEM -- PERFORMANCE REPORT")
        print(border)

        print("\n  >> SCHEDULER METRICS")
        print(f"    Requests received  : {sched['total_received']}")
        print(f"    Requests dispatched: {sched['total_dispatched']}")
        print(f"    Completed (success): {sched['total_completed']}")
        print(f"    Failed             : {sched['total_failed']}")
        print(f"    Success rate       : {sched['success_rate'] * 100:.2f}%")
        print(f"    Avg latency        : {sched['average_latency_s']:.6f}s")
        print(f"    Throughput         : {sched['throughput_rps']:.2f} req/s")

        print("\n  >> LOAD BALANCER METRICS")
        print(f"    Strategy           : {lb['strategy']}")
        print(f"    Pool size          : {lb['pool_size']}")
        print(f"    Total dispatched   : {lb['total_requests']}")
        print(f"    Successful         : {lb['successful_requests']}")
        print(f"    Failed             : {lb['failed_requests']}")
        print(f"    Success rate       : {lb['success_rate'] * 100:.2f}%")
        print(f"    Avg latency        : {lb['average_latency_s']:.6f}s")

        print("\n  >> PER-WORKER DISPATCH DISTRIBUTION")
        dispatches = lb.get("per_worker_dispatches", {})
        successes = lb.get("per_worker_successes", {})
        failures = lb.get("per_worker_failures", {})
        for wid in sorted(dispatches.keys()):
            s = successes.get(wid, 0)
            f = failures.get(wid, 0)
            print(f"    Worker {wid:>3d}  ->  dispatched={dispatches[wid]}  "
                  f"success={s}  failed={f}")

        print(f"\n{border}\n")

    # ------------------------------------------------------------------ #
    #  Private Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_request(request: Request) -> None:
        """Basic sanity checks on incoming requests.

        Raises
        ------
        ValueError
            If the request is malformed.
        """
        if not isinstance(request, Request):
            raise TypeError(
                f"Expected a Request instance, got {type(request).__name__}."
            )
        if not request.query or not request.query.strip():
            raise ValueError(
                f"Request {request.id} has an empty query — rejecting."
            )

    def _fire_callbacks(self, request: Request, response: Response) -> None:
        """Invoke all registered on-complete callbacks.

        Exceptions in callbacks are logged but never propagated so they
        cannot break the dispatch pipeline.
        """
        for cb in self._on_complete_callbacks:
            try:
                cb(request, response)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Callback %s raised %s: %s",
                    cb.__name__,
                    type(exc).__name__,
                    exc,
                )

    # ------------------------------------------------------------------ #
    #  Dunder helpers
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"Scheduler(lb={self._lb!r}, "
            f"received={self._stats.total_received}, "
            f"completed={self._stats.total_completed}, "
            f"failed={self._stats.total_failed})"
        )
