"""
workers/gpu_worker.py
=====================
GPU Worker node for the Distributed LLM Inference System.

Each GPUWorker instance represents one inference node backed by a real
Ollama LLM server.  It satisfies the WorkerNode Protocol defined in
lb/load_balancer.py, meaning it exposes:

    - id          : int
    - process()   : Request -> Response

The worker integrates the RAG retriever to enrich every prompt with
relevant context before forwarding to the LLM, implementing the
Retrieval-Augmented Generation pipeline.

Phase 3 hooks
-------------
simulate_failure() and recover() are implemented as stubs ready for
the fault-tolerance module to activate.
"""

from __future__ import annotations

import logging
import time
import threading
import os
from typing import Optional

from dotenv import load_dotenv
import requests

from common.models import Request, Response, RequestStatus
from rag.rag_retriever import RAGRetriever
from llm.inference import run_inference, check_model_available

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults — override via environment variables or constructor
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL       = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
DEFAULT_NUM_PREDICT = 60
REQUEST_TIMEOUT     = 120


class GPUWorker:
    """A single GPU-backed LLM inference worker node.

    Parameters
    ----------
    worker_id : int
        Unique identifier for this worker within the pool.
    ollama_host : str
        Base URL of the Ollama server (default: http://localhost:11434).
    model : str
        Name of the Ollama model to use for inference.
    num_predict : int
        Maximum number of tokens to generate per response.
    rag_kb_path : str
        Path to the RAG knowledge base JSON file.
    """

    def __init__(
        self,
        worker_id: int,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
        model: str = DEFAULT_MODEL,
        num_predict: int = DEFAULT_NUM_PREDICT,
        rag_kb_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag", "kb", "documents.json"),
    ) -> None:
        self.id          = worker_id
        self.ollama_host = ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model       = model
        self.num_predict = num_predict

        # Task counters (thread-safe)
        self._lock        = threading.Lock()
        self.active_tasks = 0
        self.total_tasks  = 0
        self.failed_tasks = 0

        # Phase 3 — failure simulation flag
        self._failed = False

        # RAG retriever
        try:
            self._rag = RAGRetriever(rag_kb_path)
            logger.info(
                "Worker %d initialised with RAG  |  kb_size=%d  model=%s  host=%s",
                self.id, self._rag.get_kb_size(), self.model, self.ollama_host,
            )
        except Exception as exc:
            self._rag = None
            logger.warning(
                "Worker %d: RAG retriever unavailable (%s) — running without context.",
                self.id, exc,
            )

    # ------------------------------------------------------------------
    # Core interface — called by LoadBalancer.dispatch()
    # ------------------------------------------------------------------

    def process(self, request: Request) -> Response:
        """Process a single inference request end-to-end.

        Pipeline
        --------
        1. Reject immediately if the worker is in a failed state.
        2. Retrieve RAG context for the query.
        3. Build an augmented prompt (context + query).
        4. Call Ollama via run_inference() for LLM inference.
        5. Return a Response with the generated text and latency.

        Parameters
        ----------
        request : Request
            Incoming request from the scheduler/load-balancer.

        Returns
        -------
        Response
            Contains the generated text, latency, and success flag.
        """
        if self._failed:
            logger.warning(
                "Worker %d is in FAILED state — rejecting request %d.",
                self.id, request.id,
            )
            return Response(
                id=request.id,
                request_uid=request.uid,
                result="ERROR: Worker is currently unavailable.",
                latency=0.0,
                worker_id=self.id,
                success=False,
            )

        with self._lock:
            self.active_tasks += 1
            self.total_tasks  += 1

        start          = time.perf_counter()
        request.status = RequestStatus.PROCESSING

        try:
            context = self._retrieve_context(request.query)
            prompt  = self._build_prompt(request.query, context)

            # Delegate HTTP call entirely to inference module
            inference_result = run_inference(
                prompt=prompt,
                worker_url=self.ollama_host,
                model=self.model,
                num_predict=self.num_predict,
                timeout=REQUEST_TIMEOUT,
            )

            latency = time.perf_counter() - start

            if not inference_result["success"]:
                raise RuntimeError(inference_result["error"])

            logger.debug(
                "Worker %d | Request %d done in %.3fs | tokens=%d",
                self.id, request.id, latency, inference_result["tokens"],
            )

            return Response(
                id=request.id,
                request_uid=request.uid,
                result=inference_result["response"],
                latency=latency,
                worker_id=self.id,
                success=True,
            )

        except Exception as exc:
            latency = time.perf_counter() - start
            with self._lock:
                self.failed_tasks += 1
            logger.error(
                "Worker %d | Request %d FAILED after %.3fs: %s",
                self.id, request.id, latency, exc,
            )
            return Response(
                id=request.id,
                request_uid=request.uid,
                result=f"ERROR: {exc}",
                latency=latency,
                worker_id=self.id,
                success=False,
            )

        finally:
            with self._lock:
                self.active_tasks -= 1

    # ------------------------------------------------------------------
    # RAG integration
    # ------------------------------------------------------------------

    def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant context from the knowledge base."""
        if self._rag is None:
            return ""
        try:
            return self._rag.retrieve(query, top_k=3)
        except Exception as exc:
            logger.warning("Worker %d: RAG retrieval failed: %s", self.id, exc)
            return ""

    def _build_prompt(self, query: str, context: str) -> str:
        """Construct a RAG-augmented prompt for the LLM."""
        if context:
            return (
                f"You are a helpful assistant. Use the following context to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )
        return (
            f"You are a helpful assistant.\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """Return True if the worker is reachable and not in a failed state."""
        if self._failed:
            return False
        return check_model_available(
            worker_url=self.ollama_host,
            model=self.model,
        )

    def get_load(self) -> float:
        """Return a normalised load metric for Load-Aware routing (Phase 3).

        Returns
        -------
        float
            0.0 = idle, 1.0 = fully saturated.
            Currently approximated from active_tasks (max assumed = 8).
        """
        with self._lock:
            return min(1.0, self.active_tasks / 8.0)

    # ------------------------------------------------------------------
    # Phase 3 stubs — failure simulation & recovery
    # ------------------------------------------------------------------

    def simulate_failure(self) -> None:
        """Mark this worker as failed.

        Phase 3: The fault-tolerance monitor calls this to simulate a
        node crash. All subsequent process() calls will return an
        error Response immediately without touching Ollama.
        """
        self._failed = True
        logger.warning("Worker %d: failure simulated.", self.id)

    def recover(self) -> None:
        """Restore the worker to a healthy state after a simulated failure.

        Phase 3: Called by the fault-tolerance recovery module once the
        node is confirmed reachable again.
        """
        self._failed = False
        logger.info("Worker %d: recovered and back online.", self.id)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a snapshot of this worker's runtime statistics."""
        with self._lock:
            return {
                "worker_id":    self.id,
                "active_tasks": self.active_tasks,
                "total_tasks":  self.total_tasks,
                "failed_tasks": self.failed_tasks,
                "is_healthy":   not self._failed,
                "model":        self.model,
                "ollama_host":  self.ollama_host,
            }

    def __repr__(self) -> str:
        return (
            f"GPUWorker(id={self.id}, model={self.model!r}, "
            f"total={self.total_tasks}, active={self.active_tasks}, "
            f"healthy={not self._failed})"
        )