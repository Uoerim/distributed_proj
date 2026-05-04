"""
llm/inference.py
================
Low-level LLM inference helper for the Distributed LLM Inference System.

This module provides a single public function, ``run_inference()``, that
sends a prompt to an Ollama server and returns the generated text along
with performance metadata.

It is intentionally kept thin — all orchestration (RAG, retries, stats)
lives in GPUWorker.  This module's only job is to own the HTTP call.

Why a separate module?
----------------------
* Testability — unit tests can mock ``run_inference`` without touching
  the full GPUWorker stack.
* Swappability — swapping from Ollama to vLLM or a cloud API in Phase 4
  requires changing only this file.
* Separation of concerns — network I/O is isolated from business logic.
"""

from __future__ import annotations

import logging
import time
import os
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL       = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
DEFAULT_NUM_PREDICT = 60
DEFAULT_TIMEOUT     = 300


def run_inference(
    prompt: str,
    worker_url: str = DEFAULT_OLLAMA_HOST,
    model: str = DEFAULT_MODEL,
    num_predict: int = DEFAULT_NUM_PREDICT,
    temperature: float = 0.7,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    url     = f"{worker_url.rstrip('/')}/api/generate"
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }

    start = time.perf_counter()

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data    = resp.json()
        latency = time.perf_counter() - start

        generated_text = data.get("response", "").strip()
        token_count    = data.get("eval_count", len(generated_text.split()))

        logger.debug(
            "[Inference] %.3fs | %d tokens | model=%s | host=%s",
            latency, token_count, model, worker_url,
        )

        return {
            "response": generated_text,
            "latency":  latency,
            "tokens":   token_count,
            "model":    model,
            "success":  True,
            "error":    None,
        }

    except requests.exceptions.Timeout:
        latency = time.perf_counter() - start
        msg = f"Request timed out after {timeout}s"
        logger.error("[Inference] TIMEOUT | host=%s | %.3fs", worker_url, latency)
        return _error_result(latency, model, msg)

    except requests.exceptions.ConnectionError:
        latency = time.perf_counter() - start
        msg = f"Cannot connect to Ollama at {worker_url} — is it running?"
        logger.error("[Inference] CONNECTION ERROR | host=%s", worker_url)
        return _error_result(latency, model, msg)

    except requests.exceptions.HTTPError as exc:
        latency = time.perf_counter() - start
        msg = f"HTTP error from Ollama: {exc}"
        logger.error("[Inference] HTTP ERROR | host=%s | %s", worker_url, exc)
        return _error_result(latency, model, msg)

    except Exception as exc:
        latency = time.perf_counter() - start
        msg = f"Unexpected error: {exc}"
        logger.error("[Inference] UNEXPECTED ERROR | %s", exc)
        return _error_result(latency, model, msg)

def check_model_available(
    worker_url: str = DEFAULT_OLLAMA_HOST,
    model: str = DEFAULT_MODEL,
) -> bool:
    """Return True if the model is loaded and ready on the given Ollama instance.

    Useful for health-check endpoints and startup validation.

    Parameters
    ----------
    worker_url : str
        Base URL of the Ollama server.
    model : str
        Model name to check for.
    """
    try:
        resp = requests.get(f"{worker_url.rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
        models    = [m["name"] for m in resp.json().get("models", [])]
        available = any(model in m for m in models)
        if not available:
            logger.warning(
                "[Inference] Model '%s' not found on %s. Available: %s",
                model, worker_url, models,
            )
        return available
    except Exception as exc:
        logger.error("[Inference] Health check failed for %s: %s", worker_url, exc)
        return False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _error_result(latency: float, model: str, error: str) -> Dict[str, Any]:
    """Construct a standardised error result dict."""
    return {
        "response": "",
        "latency":  latency,
        "tokens":   0,
        "model":    model,
        "success":  False,
        "error":    error,
    }