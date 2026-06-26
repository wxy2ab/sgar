"""Shared ccx cost-event emission helpers."""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def emit_cost_event(
    *, mode: str, cost_usd: float, call_count: int, tokens: int = 0,
) -> None:
    """Publish ``ccx.cost.node`` on the active v5 dispatch context."""
    from core.deepstack_v5.execution.dispatch_context import (
        current_dispatch_context,
    )

    ctx = current_dispatch_context()
    if ctx is None:
        logger.debug(
            "ccx cost event outside dispatch context: mode=%s cost=%s calls=%s",
            mode,
            cost_usd,
            call_count,
        )
        return
    if ctx.is_cancelled():
        logger.debug(
            "ccx cost event after dispatch cancellation: mode=%s node=%s",
            mode,
            ctx.node_id,
        )
        return
    ctx.emit("ccx.cost.node", {
        "run_id": ctx.run_id,
        "node_id": ctx.node_id,
        "attempt_id": ctx.attempt_id,
        "mode": mode,
        "cost_usd": cost_usd,
        "call_count": call_count,
        "tokens": int(tokens or 0),
    })


def report_cost_to_budget(*, cost_usd: float, tokens: int = 0) -> None:
    """Consume cost/tokens on the active v5 budget, if any."""
    from core.deepstack_v5.execution.dispatch_context import (
        current_dispatch_context,
    )

    ctx = current_dispatch_context()
    if ctx is None:
        logger.debug(
            "ccx budget cost outside dispatch context: cost=%s tokens=%s",
            cost_usd,
            tokens,
        )
        return
    ctx.report_cost(tokens=tokens, cost=cost_usd)
