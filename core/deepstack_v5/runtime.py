"""RuntimeV5 — single dependency-wiring entry point.

Design intent: 3 required parameters, ≤ 5 optional, hello-world fits in
22 lines. Compare to v4's RuntimeDepsV4 with 14 stores + 8 cognition
slots requiring 80 lines of boilerplate.

Default backend is SQLite (per design doc choice) — workspace path
becomes an SQLite file. Multi-process worker support is opt-in via
`worker_count > 1`; the same engine code path drives both single- and
multi-worker setups via the lease/heartbeat protocol.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


logger = logging.getLogger(__name__)

from .config import ConfigV5
from .control.budget import BudgetTracker
from .control.controller import Controller, FailureHook, ProposeInitial, ReplanHook
from .events import EventBus
from .execution.assignment import AssignmentManager
from .knowledge.claims import ClaimStore
from .knowledge.compaction import CompactionStrategy
from .persistence.db import SQLiteRuntimeDB
from .persistence.outbox import Outbox
from .persistence.stores import (
    ClaimStorePersistence,
    EventStore,
    GraphStore,
    RunStore,
    SnapshotStore,
)
from .types import Budget, Capability


EventHook = Callable[[dict[str, Any]], None]


class RuntimeV5:
    def __init__(
        self,
        *,
        capabilities: Mapping[str, Capability],
        workspace: Path,
        config: ConfigV5,
        db: SQLiteRuntimeDB,
        budget: BudgetTracker,
        controller: Controller,
        interaction_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        self.capabilities = dict(capabilities)
        self.workspace = workspace
        self.config = config
        self.db = db
        self.budget = budget
        self.controller = controller
        # Optional host callback for human-in-the-loop interaction, threaded
        # onto every per-call DispatchContext by the engine/harness so the
        # ccx ``ask_human`` tool can reach a human. ``None`` ⇒ fully autonomous.
        self.interaction_fn = interaction_fn

        # Stores (always SQLite-backed; in-memory backend is for tests
        # and bypasses RuntimeV5).
        self.run_store = RunStore(db)
        self.graph_store = GraphStore(db)
        self.event_store = EventStore(db)
        self.outbox = Outbox(db)
        self.event_bus = EventBus(db=db, event_store=self.event_store, outbox=self.outbox)

        self.assignment = AssignmentManager(
            self.graph_store,
            ttl_ms=config.lease_ttl_ms,
            heartbeat_interval_ms=config.heartbeat_interval_ms,
        )

        # Knowledge layer.
        claim_persistence = ClaimStorePersistence(db)
        self.claim_store = ClaimStore(persistence=claim_persistence)
        self.claim_store.load(limit=config.claim_store_load_limit)
        self.snapshot_store = SnapshotStore(db)
        self.compaction = config.build_compaction_strategy()
        # Phase 3: wire snapshot persistence so compaction triggers
        # write a ResumeSnapshot row alongside archiving claims. Both
        # stores already exist at this point — no chicken-and-egg.
        self.compaction.attach_stores(
            event_store=self.event_store,
            snapshot_store=self.snapshot_store,
        )
        self._wire_compaction_events()

    def _wire_compaction_events(self) -> None:
        def on_budget_warning(event: dict[str, Any]) -> None:
            try:
                plan = self.compaction.on_budget_warning(
                    event["run_id"], self.claim_store
                )
                self._publish_compaction_completed(event["run_id"], plan)
            except Exception:
                logger.warning("budget-warning compaction failed", exc_info=True)

        def on_node_completed(event: dict[str, Any]) -> None:
            payload = event.get("payload") or {}
            try:
                plan = self.compaction.on_node_succeeded(
                    event["run_id"],
                    self.claim_store,
                    tokens_added=int(payload.get("tokens", 0) or 0),
                )
                self._publish_compaction_completed(event["run_id"], plan)
            except Exception:
                logger.warning("node-completed compaction failed", exc_info=True)

        self.event_bus.subscribe(on_budget_warning, kind="budget.warning")
        self.event_bus.subscribe(on_node_completed, kind="node.completed")

    def _publish_compaction_completed(self, run_id: str, plan: Any | None) -> None:
        if plan is None or not getattr(plan, "snapshot_id", None):
            return
        self.event_bus.publish(run_id, "compaction.completed", {
            "snapshot_id": plan.snapshot_id,
            "triggered_by": plan.triggered_by,
            "snapshot_highwater_sequence": plan.snapshot_highwater_sequence,
            "archived_claims": len(plan.archive_claim_ids),
        })

    # -- factory -------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        # Required (3)
        capabilities: Mapping[str, Capability],
        workspace: Path | str,
        # Optional (≤ 5)
        propose_initial: ProposeInitial | None = None,
        budget: Budget | None = None,
        replan_hook: ReplanHook | None = None,
        failure_hook: FailureHook | None = None,
        config: ConfigV5 | None = None,
        # Power-user knobs
        llm_client: Any | None = None,
        event_hooks: Sequence[EventHook] | None = None,
        worker_count: int = 1,
        interaction_fn: Callable[[Any], Any] | None = None,
    ) -> "RuntimeV5":
        cfg = config or ConfigV5()
        if worker_count > 1:
            cfg.parallelism = max(cfg.parallelism, worker_count)
        cfg.validate()

        # Guard against the silent ":memory:" + multi-thread footgun
        # BEFORE we try to mkdir a path that doesn't exist on disk.
        # sqlite3.connect(":memory:") gives each Python thread its own
        # private in-memory DB — workers would see an empty schema. The
        # URI form ``file::memory:?cache=shared`` is required for
        # multi-thread sharing, and we leave that opt-in to the caller.
        if str(workspace) == ":memory:" and cfg.parallelism > 1:
            raise ValueError(
                "RuntimeV5: workspace ':memory:' + parallelism > 1 is not "
                "shareable across threads. Use a file path, or use the "
                "URI form 'file::memory:?cache=shared' (caller-managed)."
            )

        ws = Path(workspace)
        ws.mkdir(parents=True, exist_ok=True)
        db_path = ws / "runtime.db"
        db = SQLiteRuntimeDB(db_path)

        budget_tracker = BudgetTracker(budget if budget is not None else Budget())
        controller = Controller(
            budget=budget_tracker,
            propose_initial=propose_initial,
            replan_hook=replan_hook,
            failure_hook=failure_hook,
            llm_client=llm_client,
        )

        rt = cls(
            capabilities=capabilities,
            workspace=ws,
            config=cfg,
            db=db,
            budget=budget_tracker,
            controller=controller,
            interaction_fn=interaction_fn,
        )

        if event_hooks:
            for hook in event_hooks:
                rt.event_bus.subscribe(hook)

        return rt

    # -- API -----------------------------------------------------------------

    def engine(self):
        # Lazy import to avoid circular reference at module load.
        from .engine import EngineV5
        return EngineV5(self)

    def shutdown(self) -> None:
        try:
            self.db.close()
        except Exception:
            logger.warning(
                "RuntimeV5.shutdown: db.close() raised; ignoring",
                exc_info=True,
            )


__all__ = ["EventHook", "RuntimeV5"]
