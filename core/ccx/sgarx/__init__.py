"""SGAR Extended (sgarx) — incubation parallel of sgar.

sgarx is a fresh namespace that lets us prototype new long-horizon
governance features without disturbing the stable sgar runtime. Stage A
of sgarx is *behaviorally equivalent* to sgar: same state machine, same
operations, same trace shape — but data lives under ``.sgarx/`` so the
two never share a workspace footprint.
"""

from .runtime import SgarxRuntime
from .store import SGARX_DIR, SgarxStore

__all__ = ["SGARX_DIR", "SgarxRuntime", "SgarxStore"]
