"""
common/models.py
================
Shared data-models used across every layer of the distributed LLM system.

Provides:
    - Request  : immutable description of an incoming user query.
    - Response : result returned after a worker finishes processing.
    - RequestStatus : enum tracking the lifecycle of a request.

Design notes
------------
* ``dataclasses`` keeps the models lightweight and dependency-free.
* ``__slots__`` is intentionally *not* used so that external modules
  (e.g. monitoring dashboards) can attach ad-hoc attributes if needed.
* ``timestamp`` on Request is auto-populated via ``field(default_factory=...)``
  so the caller never has to supply it manually.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RequestStatus(Enum):
    """Lifecycle states a request can be in."""
    PENDING    = auto()
    DISPATCHED = auto()
    PROCESSING = auto()
    COMPLETED  = auto()
    FAILED     = auto()


class RoutingStrategy(Enum):
    """Load-balancing strategies recognised by the system.

    ROUND_ROBIN      – implemented in Phase 2 (current).
    LEAST_CONNECTIONS – placeholder for Phase 3.
    LOAD_AWARE        – placeholder for Phase 3.
    """
    ROUND_ROBIN       = "round_robin"
    LEAST_CONNECTIONS  = "least_connections"
    LOAD_AWARE         = "load_aware"


# ---------------------------------------------------------------------------
# Core data-classes
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """Represents an incoming user query destined for LLM inference.

    Attributes
    ----------
    id : int
        Unique, monotonically-increasing request identifier.
    query : str
        The raw user query string.
    uid : str
        A UUID4 string generated at creation time — useful for distributed
        tracing across services.
    timestamp : float
        Unix epoch when the request object was created.
    priority : int
        Optional priority level (lower = higher priority).  Reserved for
        future priority-queue scheduling.
    metadata : dict
        Arbitrary key-value bag that upstream layers can populate (e.g.
        ``{"source": "api_gateway", "user_tier": "premium"}``).
    status : RequestStatus
        Current lifecycle state of the request.
    """

    id: int
    query: str
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    status: RequestStatus = RequestStatus.PENDING


@dataclass
class Response:
    """Result payload returned by a GPU worker after inference.

    Attributes
    ----------
    id : int
        Mirrors ``Request.id`` for correlation.
    request_uid : str
        Mirrors ``Request.uid`` for distributed tracing.
    result : str
        The generated text (or error message if ``success`` is False).
    latency : float
        Wall-clock seconds the worker spent processing.
    worker_id : int
        Identifier of the GPU worker that handled the request.
    success : bool
        Whether inference completed without error.
    timestamp : float
        Unix epoch when the response was created.
    metadata : dict
        Arbitrary key-value bag for downstream consumers.
    """

    id: int
    request_uid: str
    result: str
    latency: float
    worker_id: int
    success: bool = True
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
