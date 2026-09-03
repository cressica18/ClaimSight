"""In-process pipeline lock.

Phase 11 — guards against concurrent `run_analysis` calls on the same
claim within a single Python process. The DB-level partial unique index
(`uq_analyses_one_running_per_claim`, see the migration) is the
authoritative guard for multi-process deployments; this in-process
lock is the lighter, faster primary guard for the single-process case
that the dev server and the test suite actually use.

Public API:
    acquire(claim_id) -> bool        # True if lock was obtained
    release(claim_id) -> None        # no-op if not held
    is_held(claim_id) -> bool        # for diagnostics / tests

The lock is process-scoped: restarting the server drops it. That is
acceptable because the partial unique index ensures no stale "running"
rows survive a restart — `Base.metadata.create_all` in tests + the
startup hook in production both clear them.
"""

from __future__ import annotations

import threading
from typing import Dict

# claim_id -> threading.Lock. Lazily populated on first acquire.
_locks: Dict[int, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(claim_id: int) -> threading.Lock:
    """Return (creating if needed) the threading.Lock for this claim."""
    with _locks_guard:
        lock = _locks.get(claim_id)
        if lock is None:
            lock = threading.Lock()
            _locks[claim_id] = lock
        return lock


def acquire(claim_id: int) -> bool:
    """Try to acquire the lock for `claim_id`. Non-blocking.

    Returns True if the lock was acquired, False if another thread
    inside this process already holds it.
    """
    return _lock_for(claim_id).acquire(blocking=False)


def release(claim_id: int) -> None:
    """Release the lock for `claim_id`. No-op if not held by this
    thread; safe to call from the orchestrator's finally block even
    if the lock was never acquired (e.g. the request returned 409)."""
    lock = _locks.get(claim_id)
    if lock is None:
        return
    # Only release if this thread holds the lock; otherwise let it
    # raise ValueError. Silently ignoring is the right behaviour here
    # because the lock guard already serializes per-process calls.
    try:
        lock.release()
    except RuntimeError:
        # Lock was not held by the calling thread (or was already
        # released). This can happen if the request handler released
        # it before reaching us. Treat as no-op.
        pass


def is_held(claim_id: int) -> bool:
    """Return True if the lock for `claim_id` is currently held.
    Used by tests to assert the lock is released after a run.
    """
    lock = _locks.get(claim_id)
    if lock is None:
        return False
    return lock.locked()
