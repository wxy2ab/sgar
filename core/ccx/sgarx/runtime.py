"""Validated sgarx runtime operations.

Stage A: storage backend swapped to ``.sgarx/`` via :class:`SgarxStore`.
All sgar business methods are inherited unchanged.

Stage B: two new long-horizon recovery operations are added on top of
the inherited surface — ``reopen_stage`` (undo a recent close so the
stage can be re-executed) and ``abandon_stage`` (give up on the current
stage and return to STAGE_READY). The state machine is widened in
:meth:`_set_mode` with sgarx-only edges (EXECUTION/VERIFICATION ->
STAGE_READY) so abandon can land. The reopen edge
(NEXT_STAGE_READY -> EXECUTION) is already in sgar's table, so reopen
needs no new transition.

Stage C: the context.md packet that ``start_stage`` writes is enriched
so the LLM gets exit criteria + prior handoff excerpt in one read. The
sgar baseline ``context.md`` only links to the source artifacts; sgarx
keeps that header but appends ``## Exit criteria`` (markdown checklist)
and, when applicable, ``## Previous handoff (<stage_id>)`` containing
the last paragraph of the predecessor stage's ``handoff.md``.

Stage D: missions optionally bind to an exit criterion via the
``target_criterion`` manifest field. When such a mission is completed
during an active stage whose spec lists that criterion, sgarx upserts
an *auto-pending* record into ``verification.json`` (passed=False with
``auto_pending: true`` + the source mission id). Close-stage will reject
it until a human/agent runs the explicit ``verify --pass`` to overwrite.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..sgar.models import (
    ProjectMode,
    ProjectState,
    SgarError,
    StageRecord,
    StageStatus,
)
from ..sgar.missions import mission_manifest_path, validate_mission_id
from ..sgar.runtime import ALLOWED_TRANSITIONS, SgarRuntime
from ..sgar.store import utc_now
from ..sgar.tracing import SgarTracer
from ..sgar.validation import parse_exit_criteria
from .store import SgarxStore


# Sgarx-local transition extensions. Additive on top of sgar's table —
# we never remove an existing edge, only add ones sgar refuses.
SGARX_EXTRA_TRANSITIONS: dict[str, set[str]] = {
    ProjectMode.EXECUTION.value: {
        ProjectMode.STAGE_READY.value,
        ProjectMode.ROADMAP.value,
    },
    ProjectMode.VERIFICATION.value: {
        ProjectMode.STAGE_READY.value,
        ProjectMode.ROADMAP.value,
    },
}


# Stage status string written by abandon_stage. We deliberately do not
# touch :class:`StageStatus` (sgar enum) — sgarx records this literal in
# the StageRecord.status string field.
STAGE_STATUS_ABANDONED = "abandoned"


class SgarxRuntime(SgarRuntime):
    def __init__(
        self,
        cwd: str | Path = ".",
        session_id: str | None = None,
        *,
        run_criterion_checks: bool = False,
        criterion_check_timeout_s: float = 120.0,
    ) -> None:
        self.store = SgarxStore(cwd, session_id=session_id)
        self.tracer = SgarTracer(self.store)
        # See SgarRuntime.__init__ — P2 opt-in check execution (default off).
        # SgarxRuntime does not call super().__init__ (it swaps in
        # SgarxStore), so these must be set here too; record_verification /
        # close_stage are inherited and read them.
        self.run_criterion_checks = run_criterion_checks
        self.criterion_check_timeout_s = criterion_check_timeout_s

    def _set_mode(
        self,
        state: ProjectState,
        new_mode: str,
        *,
        allow_extra: bool = False,
    ) -> None:
        allowed = set(ALLOWED_TRANSITIONS.get(state.mode, set()))
        if allow_extra:
            allowed |= SGARX_EXTRA_TRANSITIONS.get(state.mode, set())
        if new_mode not in allowed:
            raise SgarError(
                f"invalid SGARX transition: {state.mode} -> {new_mode}"
            )
        state.mode = new_mode

    # ------------------------------------------------------------------ #
    # Stage B: reopen / abandon
    # ------------------------------------------------------------------ #

    def reopen_stage(self, stage_id: str, *, reason: str) -> ProjectState:
        """Undo a recent close so the stage can be re-executed.

        Constraints (Stage B): only the most-recently-closed stage may be
        reopened, and only while the project is parked at NEXT_STAGE_READY.
        Any prior verification.json is archived to a sibling
        ``verification.superseded.<utc>.json`` with a top-level
        ``superseded: true`` marker, then removed so close_stage will
        require a fresh verification.
        """
        if not reason or not reason.strip():
            raise SgarError("reopen-stage requires --reason")
        reason = reason.strip()
        with self.tracer.operation(
            "reopen_stage",
            inputs={"stage_id": stage_id, "reason": reason},
        ) as trace:
            state = self._refresh_flags()
            if state.mode != ProjectMode.NEXT_STAGE_READY.value:
                raise SgarError(
                    f"cannot reopen {stage_id}; project mode is {state.mode}, "
                    f"expected {ProjectMode.NEXT_STAGE_READY.value}"
                )
            if state.last_closed_stage_id != stage_id:
                raise SgarError(
                    f"cannot reopen {stage_id}; most recent closed stage is "
                    f"{state.last_closed_stage_id!r}"
                )

            archived_path = self._archive_verification(stage_id, reason=reason)

            record = state.stages.get(stage_id) or StageRecord(stage_id=stage_id)
            record.status = StageStatus.EXECUTION.value
            record.closed_at = None
            state.stages[stage_id] = record
            if stage_id in state.closed_stage_ids:
                state.closed_stage_ids.remove(stage_id)
            state.current_stage_id = stage_id
            state.next_stage_id = stage_id
            state.last_closed_stage_id = None
            self._set_mode(state, ProjectMode.EXECUTION.value)
            self.store.write_state(state)

            artifacts = self._artifacts(("state", self.store.state_path))
            if archived_path is not None:
                artifacts.append(
                    self.tracer.artifact(archived_path, role="superseded_verification")
                )
            trace["artifacts"] = artifacts
            return state

    def abandon_stage(self, stage_id: str, *, reason: str) -> ProjectState:
        """Give up on the current stage and return to STAGE_READY.

        Constraints (Stage B): the stage must be the current one and the
        project must be in EXECUTION or VERIFICATION. Any partial
        verification.json is archived (same scheme as reopen) so the next
        attempt starts from a clean slate. The stage record's status is
        written as ``"abandoned"`` (a sgarx-local extension of the sgar
        StageStatus enum).
        """
        if not reason or not reason.strip():
            raise SgarError("abandon-stage requires --reason")
        reason = reason.strip()
        with self.tracer.operation(
            "abandon_stage",
            inputs={"stage_id": stage_id, "reason": reason},
        ) as trace:
            state = self._refresh_flags()
            if state.current_stage_id != stage_id:
                raise SgarError(
                    f"cannot abandon {stage_id}; current stage is "
                    f"{state.current_stage_id!r}"
                )
            if state.mode not in {
                ProjectMode.EXECUTION.value,
                ProjectMode.VERIFICATION.value,
            }:
                raise SgarError(
                    f"cannot abandon {stage_id}; project mode is {state.mode}, "
                    f"expected execution or verification"
                )

            archived_path = self._archive_verification(stage_id, reason=reason)

            record = state.stages.get(stage_id) or StageRecord(stage_id=stage_id)
            record.status = STAGE_STATUS_ABANDONED
            record.closed_at = utc_now()
            state.stages[stage_id] = record
            state.current_stage_id = None
            self._set_mode(state, ProjectMode.STAGE_READY.value, allow_extra=True)
            self.store.write_state(state)

            artifacts = self._artifacts(("state", self.store.state_path))
            if archived_path is not None:
                artifacts.append(
                    self.tracer.artifact(archived_path, role="superseded_verification")
                )
            trace["artifacts"] = artifacts
            return state

    # ------------------------------------------------------------------ #
    # Stage C: enriched context packet
    # ------------------------------------------------------------------ #

    def _write_stage_execution_files(self, stage_id: str, criteria: list) -> None:
        # Write the sgar baseline files first (plan.md, tasks.md, context.md
        # placeholder), then overwrite context.md with the enriched packet.
        super()._write_stage_execution_files(stage_id, criteria)
        self.store.write_text(
            self.store.stage_context_path(stage_id),
            self._build_enriched_context(stage_id, criteria),
        )

    def _build_enriched_context(self, stage_id: str, criteria: list) -> str:
        lines = [
            f"# Context Packet: {stage_id}",
            "",
            f"- Blueprint: `{self.store.blueprint_path}`",
            f"- Roadmap: `{self.store.roadmap_path}`",
            f"- Stage spec: `{self.store.stage_spec_path(stage_id)}`",
            f"- Blueprint hash: `{self.store.file_hash(self.store.blueprint_path)}`",
            f"- Roadmap hash: `{self.store.file_hash(self.store.roadmap_path)}`",
            "",
            "## Exit criteria",
            "",
        ]
        if criteria:
            for criterion in criteria:
                lines.append(
                    f"- [ ] {criterion.criterion_id}: {criterion.description}"
                )
        else:
            lines.append("_(no exit criteria parsed from spec — fix `spec.md` before proceeding)_")
        lines.append("")

        excerpt = self._previous_handoff_excerpt(stage_id)
        if excerpt is not None:
            lines += [
                f"## Previous handoff ({excerpt['stage_id']})",
                "",
                excerpt["text"],
                "",
            ]
        return "\n".join(lines)

    def _previous_handoff_excerpt(self, stage_id: str) -> dict | None:
        """Locate the most recently closed stage that isn't ``stage_id``
        and return ``{stage_id, text}`` where ``text`` is the last
        non-empty paragraph of its ``handoff.md``. Returns None if no
        prior closed stage exists or its handoff is missing/empty."""
        state = self.store.load_state()
        candidates = [sid for sid in state.closed_stage_ids if sid != stage_id]
        if not candidates:
            return None
        prev = candidates[-1]
        handoff_path = self.store.handoff_path(prev)
        if not handoff_path.exists():
            return None
        text = handoff_path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return None
        return {"stage_id": prev, "text": paragraphs[-1]}

    # ------------------------------------------------------------------ #
    # Stage D: mission target_criterion auto-binding
    # ------------------------------------------------------------------ #

    def create_mission(
        self,
        *,
        mission_id: str,
        kind: str,
        objective: str,
        input_paths: list,
        expected_outputs: list[str],
        allowed_scope: list[str] | None = None,
        target_criterion: str | None = None,
    ) -> dict:
        # Validate BEFORE creating anything on disk: raising after
        # ``super().create_mission()`` used to leave an orphaned active
        # mission whose id then blocked the corrected retry.
        if target_criterion is not None:
            target_criterion = target_criterion.strip()
            if not target_criterion:
                raise SgarError("target_criterion cannot be blank")
        manifest = super().create_mission(
            mission_id=mission_id,
            kind=kind,
            objective=objective,
            input_paths=input_paths,
            expected_outputs=expected_outputs,
            allowed_scope=allowed_scope,
        )
        if target_criterion is not None:
            manifest_path = self.store.missions_root / mission_id / "manifest.json"
            data = self.store.read_json(manifest_path)
            data["target_criterion"] = target_criterion
            # store.write_json = atomic temp+rename, consistent format
            # (sort_keys + trailing newline) with every other manifest.
            self.store.write_json(manifest_path, data)
            manifest["target_criterion"] = target_criterion
        return manifest

    def complete_mission(
        self,
        mission_id: str,
        *,
        result_path,
    ) -> dict:
        mission_id = validate_mission_id(mission_id)
        manifest_path = mission_manifest_path(self.store, mission_id)
        pre_manifest = self.store.read_json(manifest_path)
        target = pre_manifest.get("target_criterion")
        if not isinstance(target, str) or not target.strip():
            return super().complete_mission(mission_id, result_path=result_path)
        target = target.strip()

        # Compute trace inputs first so any skip path can include the
        # decision reason — SgarTracer only persists what's in `inputs`
        # plus what we set on `trace["artifacts"]`. Phantom keys like
        # `trace["skipped_reason"]` would be silently dropped.
        trace_inputs: dict = {
            "mission_id": mission_id,
            "target_criterion": target,
        }
        state = self.store.load_state()
        stage_id = state.current_stage_id
        if not stage_id:
            trace_inputs["skipped_reason"] = (
                "no current stage; pending record not written"
            )
            manifest = super().complete_mission(mission_id, result_path=result_path)
            with self.tracer.operation(
                "mission_pending_verification",
                inputs=trace_inputs,
            ) as trace:
                trace["artifacts"] = self._artifacts(
                    ("mission_manifest", manifest_path),
                )
            return manifest

        spec_path = self.store.stage_spec_path(stage_id)
        if not spec_path.exists():
            trace_inputs["skipped_reason"] = f"stage spec missing: {spec_path}"
            manifest = super().complete_mission(mission_id, result_path=result_path)
            with self.tracer.operation(
                "mission_pending_verification",
                inputs=trace_inputs,
            ) as trace:
                trace["artifacts"] = self._artifacts(
                    ("mission_manifest", manifest_path),
                )
            return manifest

        criteria = parse_exit_criteria(spec_path.read_text(encoding="utf-8"))
        known = {c.criterion_id for c in criteria}
        if target not in known:
            trace_inputs["skipped_reason"] = (
                f"target_criterion {target!r} is not an exit criterion of "
                f"current stage {stage_id!r} (known: {sorted(known)})"
            )
            manifest = super().complete_mission(mission_id, result_path=result_path)
            with self.tracer.operation(
                "mission_pending_verification",
                inputs=trace_inputs,
            ) as trace:
                trace["artifacts"] = self._artifacts(
                    ("mission_manifest", manifest_path),
                )
            return manifest

        verification_path = self.store.verification_json_path(stage_id)
        if verification_path.exists():
            self.store.read_json(verification_path)
        manifest = super().complete_mission(mission_id, result_path=result_path)
        with self.tracer.operation(
            "mission_pending_verification",
            inputs=trace_inputs,
        ) as trace:
            self._upsert_pending_verification(
                stage_id=stage_id,
                criterion_id=target,
                mission_id=mission_id,
            )
            trace["artifacts"] = self._artifacts(
                ("verification_json", verification_path),
                ("mission_manifest", manifest_path),
            )
        return manifest

    def _upsert_pending_verification(
        self,
        *,
        stage_id: str,
        criterion_id: str,
        mission_id: str,
    ) -> Path | None:
        """Insert (or replace) a pending CriterionResult for ``criterion_id``
        in ``verification.json``. The record is written with ``passed=False``
        plus a sgarx-specific ``auto_pending`` marker so tooling can tell
        it apart from a real failure. Returns None — verification.json is
        rewritten in place; nothing is archived.

        Single-writer contract (see core/ccx/docs/supervised/
        sgarx_concurrency_2026-06-24.md): this is the lock-free read-modify-write
        that, under *emergent* concurrency (two sibling sgarx nodes on one
        cwd+session), could silently lose a co-racer's update (P1) or resurrect
        a record for an abandoned stage (P5), with the loser's trace lying
        ``completed``. The write below is therefore guarded so the loser is
        *loud and honest* instead of silent:
          - CAS on the verification.json generation captured before the read →
            a concurrent ``complete_mission`` that wrote it first is detected
            (P1); we abort with SgarError rather than clobber its record.
          - precondition "this stage is still current" → a concurrent
            ``abandon``/transition that revoked the stage is detected (P5); we
            refuse to write an orphan auto-pending record.
        Both raise SgarError, which ``complete_mission``'s tracer records as a
        FAILED ``mission_pending_verification`` (no completed+artifact lie) and
        the dispatch catches. On the single-writer path the token matches and
        the stage stays current, so this is byte-for-byte equivalent to the
        old unguarded ``write_json``."""
        path = self.store.verification_json_path(stage_id)
        # Capture the generation BEFORE the read so the guarded write can tell
        # a concurrent writer apart from our own no-op. None == absent at read.
        token_before = self.store.stat_token(path)
        if path.exists():
            data = self.store.read_json(path)
        else:
            data = {"stage_id": stage_id, "results": [], "notes": ""}
        results = data.get("results")
        if not isinstance(results, list):
            results = []
        evidence = (
            f"auto-pending from mission {mission_id}: "
            f"explicit verify --pass required to close stage"
        )
        new_record = {
            "criterion_id": criterion_id,
            "passed": False,
            "evidence": evidence,
            "auto_pending": True,
            "mission_id": mission_id,
            "recorded_at": utc_now(),
        }
        replaced = False
        for index, existing in enumerate(results):
            if isinstance(existing, dict) and existing.get("criterion_id") == criterion_id:
                results[index] = new_record
                replaced = True
                break
        if not replaced:
            results.append(new_record)
        data["results"] = results
        data.setdefault("stage_id", stage_id)
        data.setdefault("notes", "")
        # Atomic, compare-and-swap-guarded write via the store. A torn write
        # would brick every subsequent verification read for the stage; a
        # silent clobber would drop a co-racer's pending record (P1); a write
        # for a no-longer-current stage would resurrect an orphan (P5). The
        # CAS + precondition turn both races into a loud SgarError instead.
        self.store.write_json_cas(
            path,
            data,
            expected=token_before,
            precondition=lambda: (
                self.store.load_state().current_stage_id == stage_id
            ),
        )
        return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _archive_verification(self, stage_id: str, *, reason: str) -> Path | None:
        """If a verification.json exists for ``stage_id``, copy it to a
        timestamped ``verification.superseded.<utc>.json`` sibling with a
        top-level ``superseded: true`` marker, then delete the original.
        Returns the archived path, or None if there was nothing to archive.
        """
        source = self.store.verification_json_path(stage_id)
        try:
            raw_text = source.read_text(encoding="utf-8")
        except FileNotFoundError:
            # Nothing to archive: either it never existed, or a concurrent
            # same-stage archiver already took it. EAFP read replaces an
            # exists()-then-read TOCTOU that crashed the loser of the race
            # (byte-equivalent single-threaded — absent still returns None).
            return None
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Preserve the original bytes — a corrupt file is exactly
            # the case where discarding the content loses evidence.
            data = {"unparseable_original": raw_text}
        timestamp = utc_now().replace(":", "").replace("-", "")
        # Same-second reopen+abandon (sequentially OR from two concurrent
        # SgarxRuntime instances on one workspace) would collide on the
        # timestamped name. The old ``while archived.exists()`` probe was a
        # TOCTOU: two racers both see the name free and clobber each other.
        # Reserve the name atomically via a non-globbed ``.reserving`` sentinel
        # (O_CREAT|O_EXCL), then write the real archive through the store's
        # atomic temp+rename so the canonical ``verification.superseded.*.json``
        # name only ever appears FULLY-FORMED to a glob+parse reader (the
        # sentinel is never globbed). On any write failure the ``finally``
        # removes the sentinel, leaving no reader-visible 0-byte orphan.
        # Single-writer behaviour is unchanged — the first name is always free,
        # so the canonical ``verification.superseded.<ts>.json`` is produced.
        archived, sentinel = self._reserve_archive_name(source, timestamp)
        archived_payload = {
            **data,
            "superseded": True,
            "superseded_at": utc_now(),
            "superseded_reason": reason,
        }
        try:
            self.store.write_json(archived, archived_payload)
        finally:
            sentinel.unlink(missing_ok=True)
        # missing_ok: under a concurrent same-stage archive the source may
        # already be gone (the other racer unlinked it). The exists() check at
        # the top of this method is a TOCTOU; tolerate the loss here so the op
        # degrades gracefully instead of raising an uncaught FileNotFoundError
        # (the dispatch only catches SgarError). The source content was
        # already preserved in an archive above, so nothing is lost.
        source.unlink(missing_ok=True)
        # The companion verification.md is human-facing and stale once the
        # JSON has been archived; remove it too so the next verification
        # writes a fresh one.
        self.store.verification_md_path(stage_id).unlink(missing_ok=True)
        return archived

    @staticmethod
    def _reserve_archive_name(source: Path, timestamp: str) -> tuple[Path, Path]:
        """Atomically reserve a unique ``verification.superseded.*`` archive
        name for ``source``, returning ``(final_path, sentinel_path)``.

        The sentinel is ``<final>.reserving`` — created with O_CREAT|O_EXCL so
        two concurrent archivers cannot reserve the same name, and suffixed
        (``.reserving``) so it does NOT match readers' ``verification.superseded.
        *.json`` glob. The caller writes ``final_path`` via the store's atomic
        rename (canonical name appears fully-formed) and removes the sentinel.
        The first candidate is the canonical ``verification.superseded.<ts>.json``;
        a name already taken by a live sentinel OR an existing final archive
        bumps a numeric suffix, matching the prior sequential naming scheme."""
        suffix: int | None = None
        while True:
            name = (
                f"verification.superseded.{timestamp}.json"
                if suffix is None
                else f"verification.superseded.{timestamp}.{suffix}.json"
            )
            final = source.with_name(name)
            sentinel = source.with_name(name + ".reserving")
            try:
                fd = os.open(
                    sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
                )
            except FileExistsError:
                # Another in-flight archiver holds this name.
                suffix = 2 if suffix is None else suffix + 1
                continue
            os.close(fd)
            if final.exists():
                # An already-completed archive owns this name (sequential
                # same-second case) — release the sentinel and try the next.
                sentinel.unlink(missing_ok=True)
                suffix = 2 if suffix is None else suffix + 1
                continue
            return final, sentinel

    # ------------------------------------------------------------------ #
    # Archive retention (opt-in; default never runs)
    # ------------------------------------------------------------------ #

    def gc_superseded_archives(
        self,
        stage_id: str,
        *,
        keep_last: int | None = None,
        keep_within_s: float | None = None,
    ) -> list[Path]:
        """Prune a stage's ``verification.superseded.*.json`` audit trail.

        Every reopen/abandon archives one verification.json that is **never**
        GC'd by design, so a very-long-lived workspace accumulates them
        unbounded (the durability soak quantifies ~1 per recovery cycle). This
        is the **opt-in** retention valve: it does NOTHING unless explicitly
        called with a policy, so the audit trail stays complete by default. It
        is byte-equivalent to the prior runtime (a pure additive operator/
        maintenance method; no recovery path invokes it).

        Retention (give either or both — an archive is KEPT if ANY given policy
        keeps it; with neither, this is a no-op returning ``[]``):
          ``keep_last``     — keep the N most-recent archives, prune older.
          ``keep_within_s`` — keep archives whose ``superseded_at`` is within N
                              seconds of the newest archive's, prune older.

        Only the timestamped archive siblings are touched; the live
        ``verification.json`` is never a glob match and is never removed.
        Returns the list of removed paths (oldest-first)."""
        if keep_last is not None and keep_last < 0:
            raise SgarError("gc_superseded_archives: keep_last must be >= 0")
        if keep_within_s is not None and keep_within_s < 0:
            raise SgarError("gc_superseded_archives: keep_within_s must be >= 0")
        stage_dir = self.store.stage_dir(stage_id)
        if not stage_dir.exists():
            return []
        archives = list(stage_dir.glob("verification.superseded.*.json"))
        if not archives or (keep_last is None and keep_within_s is None):
            return []
        # Oldest -> newest. Sort by recorded superseded_at, then by name so the
        # same-second collision suffixes (.<n>) order deterministically.
        ordered = sorted(archives, key=lambda p: (self._archive_superseded_at(p), p.name))
        keep: set[Path] = set()
        if keep_last is not None and keep_last > 0:
            keep |= set(ordered[-keep_last:])
        if keep_within_s is not None:
            newest = self._archive_superseded_at(ordered[-1])
            cutoff = newest - timedelta(seconds=keep_within_s)
            keep |= {p for p in ordered if self._archive_superseded_at(p) >= cutoff}
        removed: list[Path] = []
        for path in ordered:
            if path not in keep:
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                removed.append(path)
        return removed

    @staticmethod
    def _archive_superseded_at(path: Path) -> datetime:
        """Timezone-aware archive timestamp for retention ordering: the
        ``superseded_at`` field, falling back to file mtime when it is
        missing/unparseable/unreadable (so a hand-edited archive still sorts)."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            stamped = data.get("superseded_at")
            if isinstance(stamped, str) and stamped:
                dt = datetime.fromisoformat(stamped)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (OSError, ValueError):
            pass
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return datetime.fromtimestamp(0, tz=timezone.utc)
