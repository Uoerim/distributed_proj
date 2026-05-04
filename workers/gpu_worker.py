"""
workers/gpu_worker.py
=====================
GPU Worker Node for the Distributed LLM Inference System.

Each ``GPUWorker`` instance simulates a GPU-backed compute node that:

1. Receives a ``Request`` from the Load Balancer.
2. Retrieves contextual data via the RAG pipeline.
3. Runs LLM inference augmented with the retrieved context.
4. Returns a ``Response`` with the generated text and timing metadata.

Design Decisions
----------------
* **Simulated latency** — ``time.sleep`` models real GPU inference time.
  Replace with actual CUDA / PyTorch calls in production.
* **Thread safety** — An ``active_connections`` counter (guarded by a
  ``threading.Lock``) lets the Load Balancer's *Least Connections*
  strategy (Phase 3) query how busy each worker is.
* **Load metric** — ``get_load()`` returns a normalised float [0.0, 1.0]
  based on active connections vs. a configurable ``max_concurrent``
  capacity.  This feeds into *Load-Aware* routing (Phase 3).
* **Failure simulation** — ``failure_rate`` can be set > 0.0 during
  testing to exercise the Scheduler's error-handling paths.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Dict, Optional

from common.models import Request, Response

logger = logging.getLogger(__name__)


class GPUWorker:
    """Simulates a GPU-backed inference worker node.

    Parameters
    ----------
    id : int
        Unique numeric identifier for this worker.
    max_concurrent : int
        Maximum number of requests the worker can handle simultaneously.
        Used to compute the load metric.
    failure_rate : float
        Probability in [0.0, 1.0] that a request will fail (for testing).
    """

    def __init__(
        self,
        id: int,
        max_concurrent: int = 10,
        failure_rate: float = 0.0,
    ) -> None:
        self.id = id
        self.max_concurrent = max_concurrent
        self.failure_rate = failure_rate

        # Concurrency tracking
        self._active_connections = 0
        self._lock = threading.Lock()

        # Cumulative stats
        self._total_processed = 0
        self._total_failures = 0

        logger.info("GPUWorker %d initialised  |  max_concurrent=%d", id, max_concurrent)

    # ------------------------------------------------------------------ #
    #  Core Processing
    # ------------------------------------------------------------------ #

    def process(self, request: Request) -> Response:
        """Execute RAG retrieval + LLM inference for *request*.

        Parameters
        ----------
        request : Request
            The incoming user query.

        Returns
        -------
        Response
            Generated result with latency and worker metadata.

        Raises
        ------
        RuntimeError
            If the simulated failure trigger fires (controlled by
            ``failure_rate``).
        """
        with self._lock:
            self._active_connections += 1

        start = time.time()

        logger.debug(
            "[Worker %d] Processing request %d  (uid=%s)",
            self.id,
            request.id,
            request.uid[:8],
        )

        try:
            # -- Simulated failure ----------------------------------------
            if self.failure_rate > 0 and random.random() < self.failure_rate:
                raise RuntimeError(
                    f"Simulated GPU fault on Worker {self.id}"
                )

            # -- RAG Step -------------------------------------------------
            from rag.retriever import retrieve_context
            context = retrieve_context(request.query)

            # -- LLM Inference Step ---------------------------------------
            from llm.inference import run_llm
            result = run_llm(request.query, context)

            latency = time.time() - start

            with self._lock:
                self._total_processed += 1

            logger.debug(
                "[Worker %d] Request %d completed in %.4fs",
                self.id,
                request.id,
                latency,
            )

            return Response(
                id=request.id,
                request_uid=request.uid,
                result=result,
                latency=latency,
                worker_id=self.id,
                success=True,
            )

        except Exception as exc:
            latency = time.time() - start

            with self._lock:
                self._total_failures += 1

            logger.error(
                "[Worker %d] Request %d FAILED after %.4fs: %s",
                self.id,
                request.id,
                latency,
                exc,
            )
            raise

        finally:
            with self._lock:
                self._active_connections -= 1

    # ------------------------------------------------------------------ #
    #  Load & Health Metrics  (Phase 3 integration points)
    # ------------------------------------------------------------------ #

    def get_load(self) -> float:
        """Return normalised load [0.0 = idle, 1.0 = saturated].

        Used by the Load-Aware routing strategy (Phase 3).
        """
        with self._lock:
            return min(self._active_connections / max(self.max_concurrent, 1), 1.0)

    @property
    def active_connections(self) -> int:
        """Number of requests currently being processed."""
        with self._lock:
            return self._active_connections

    def get_stats(self) -> Dict[str, Any]:
        """Return a snapshot of this worker's metrics."""
        with self._lock:
            return {
                "worker_id": self.id,
                "active_connections": self._active_connections,
                "load": round(self.get_load(), 4),
                "total_processed": self._total_processed,
                "total_failures": self._total_failures,
                "max_concurrent": self.max_concurrent,
            }

    # ------------------------------------------------------------------ #
    #  Dunder helpers
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"GPUWorker(id={self.id}, "
            f"active={self._active_connections}, "
            f"processed={self._total_processed})"
        )
