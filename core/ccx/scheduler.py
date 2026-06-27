"""Scheduled / polling supervisor for ccx — wake on an interval or at a time
point and drive a fresh bounded agent run each tick.

ccx is run-to-completion: ``CodeAgent.run`` drives a v5 DAG once. This module
adds the missing *outer* loop for monitoring-type work ("every 5 min check X;
alert on anomaly", "at 09:30 each trading day run the morning routine") without
touching the run path. Each tick is a fresh ``CodeAgent.run_sync`` — the bounded
run ccx already does well — and state bridges between ticks through the EXISTING
primitives: the single-chain resume metadata
(``RESUME_PREVIOUS_RUN_METADATA_KEY``) and cross-run memory. There is no
long-lived sleeping run holding v5 leases/DB handles between ticks; between
ticks only this supervisor thread sleeps.

Design notes:

* ``ScheduleSpec.next_fire_at`` is the only trigger seam — pure and easily
  unit-tested with frozen ``datetime`` values. Two kinds ship: ``interval``
  (every N seconds) and ``daily`` (one or more local ``HH:MM`` points, with an
  optional weekday filter). A future ``cron`` kind can be added behind the same
  seam without touching the loop.
* The sleep *between* ticks is sliced (``_interruptible_sleep_until``) so an
  interrupt lands within ~1s **while the supervisor is waiting** — mirroring
  ``watch.py``'s follow loop and ``llm_monitor.run_monitor``'s injectable
  ``sleeper``. The CLI (``main``) installs SIGINT+SIGTERM handlers that raise
  ``KeyboardInterrupt`` so *both* signals reach the clean rc-0 exit even for a
  daemon launched in the background (where SIGINT is otherwise inherited as
  SIG_IGN); the Python API leaves the host's handlers untouched. A signal that
  arrives **mid-tick** — while ``agent.run_sync`` is blocking — only takes
  effect after that tick's blocking call returns (or its node wall-clock
  fires): it is *not* a ~1s guarantee for an already-running tick.
* The agent stops the schedule in-band: it writes a memory entry tagged
  ``ccx_schedule_stop`` (durable) or emits the ``[[SCHEDULE_STOP]]`` sentinel in
  its final text (cheap). Doing neither = keep watching (the default).
* Idempotency is the agent's job (the supervisor adds no side effects of its
  own); each tick is a single drive so there is no within-tick double-apply.

The CLI (``python -m core.ccx.scheduler``) is a thin wrapper over
``run_schedule`` and is gated behind ``CCX_SCHEDULE_ENABLE`` so an accidental
invocation cannot spin a forever-loop.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from core.deepstack_v5.memory.resume import RESUME_PREVIOUS_RUN_METADATA_KEY

logger = logging.getLogger(__name__)


# In-band stop signals from the agent (see _agent_requested_stop).
SCHEDULE_STOP_TAG = "ccx_schedule_stop"
SCHEDULE_STOP_SENTINEL = "[[SCHEDULE_STOP]]"

# Default memory tag stamped on every tick so recall/summaries are scoped to
# the monitoring history rather than mixing with unrelated runs.
MONITORING_TAG = "monitoring"

# Metadata key carrying per-tick bookkeeping (visible to the agent + tests).
SCHEDULE_METADATA_KEY = "ccx_schedule"

# CLI gate (default OFF for the daemon surface; the Python API is always usable).
_ENABLE_FLAG_ENV = "CCX_SCHEDULE_ENABLE"
_TRUTHY = frozenset({"1", "true", "on", "yes"})

_WEEKDAY_NAMES = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}


def schedule_enabled() -> bool:
    """Whether the CLI is allowed to start a scheduling loop (default OFF)."""
    raw = os.environ.get(_ENABLE_FLAG_ENV)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY


def _now_local() -> datetime:
    """Current local wall-clock time as an aware datetime."""
    return datetime.now().astimezone()


# --------------------------------------------------------------------------- #
# Schedule spec — the trigger seam
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class ScheduleSpec:
    """When to fire. ``next_fire_at`` is pure and the only trigger seam.

    * ``interval``: every ``interval_seconds``. By default the next wake is
      ``now + interval`` (computed *after* each tick returns), so a slow tick
      self-corrects forward and missed slots are skipped — never a burst. With
      ``catch_up=True`` the cadence is anchored to the previous fire so an
      overrun fires the next slot immediately (preserving the grid).
    * ``daily``: the next future ``HH:MM`` among ``daily_times`` (local time),
      restricted to ``weekdays`` when set. A missed daily slot (process down) is
      skipped — monitoring wants current state, not replay.
    """

    kind: str = "interval"
    interval_seconds: float | None = None
    daily_times: tuple[str, ...] = ()
    weekdays: frozenset[int] | None = None
    catch_up: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("interval", "daily"):
            raise ValueError(f"ScheduleSpec: unknown kind {self.kind!r}")
        if self.kind == "interval":
            if not self.interval_seconds or self.interval_seconds <= 0:
                raise ValueError(
                    "ScheduleSpec(kind='interval') requires interval_seconds > 0"
                )
        else:  # daily
            if not self.daily_times:
                raise ValueError(
                    "ScheduleSpec(kind='daily') requires at least one daily_times HH:MM"
                )
            for hhmm in self.daily_times:
                _parse_hhmm(hhmm)  # validate eagerly (raises on malformed)
            if self.weekdays is not None and not self.weekdays:
                raise ValueError("ScheduleSpec: weekdays set must be non-empty or None")

    def next_fire_at(
        self, now: datetime, last_fire: datetime | None = None
    ) -> datetime:
        """Return the next fire time strictly after ``now``.

        ``last_fire`` (the previous wake) is only consulted for
        ``interval`` + ``catch_up`` grid anchoring; it is ignored otherwise.
        """
        if self.kind == "interval":
            assert self.interval_seconds is not None
            step = timedelta(seconds=self.interval_seconds)
            if self.catch_up and last_fire is not None:
                return last_fire + step
            return now + step
        return self._next_daily(now)

    def _next_daily(self, now: datetime) -> datetime:
        times = sorted(_parse_hhmm(t) for t in self.daily_times)
        # Search today and the next 7 days for the earliest valid slot > now.
        for day_offset in range(0, 8):
            day = (now + timedelta(days=day_offset)).date()
            if self.weekdays is not None and day.weekday() not in self.weekdays:
                continue
            for hour, minute in times:
                # Resolve the local UTC offset for THIS candidate's own date
                # (DST-aware, via the system local zone) rather than freezing
                # ``now``'s current fixed offset onto a future date. ``_now_local``
                # carries a fixed-offset tzinfo, so reusing it would make a daily
                # slot computed across a DST boundary drift by the offset delta
                # (e.g. fire 09:30 at the wrong wall-clock hour for ~1 day after
                # a transition). A naive local time → ``astimezone()`` picks the
                # right offset for that calendar day.
                candidate = datetime(
                    day.year, day.month, day.day, hour, minute,
                ).astimezone()
                if candidate > now:
                    return candidate
        # Unreachable for any non-empty weekday set (a valid weekday recurs
        # within 7 days); guard anyway.
        raise ValueError("ScheduleSpec(kind='daily'): no valid slot within 8 days")


def _parse_hhmm(value: str) -> tuple[int, int]:
    text = str(value or "").strip()
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"invalid HH:MM time {value!r} (expected like '09:30')")
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"invalid HH:MM time {value!r}") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"out-of-range HH:MM time {value!r}")
    return hour, minute


def _parse_weekdays(spec: str | None) -> frozenset[int] | None:
    """Parse ``mon-fri`` / ``mon,wed,fri`` / ``0-4`` into a weekday set."""
    if not spec:
        return None
    text = spec.strip().lower()
    out: set[int] = set()

    def _one(token: str) -> int:
        token = token.strip()
        if token in _WEEKDAY_NAMES:
            return _WEEKDAY_NAMES[token]
        if token.isdigit():
            n = int(token)
            if 0 <= n <= 6:
                return n
        raise ValueError(f"invalid weekday token {token!r}")

    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = _one(lo_s), _one(hi_s)
            rng = range(lo, hi + 1) if lo <= hi else list(range(lo, 7)) + list(range(0, hi + 1))
            out.update(rng)
        else:
            out.add(_one(chunk))
    return frozenset(out) if out else None


# --------------------------------------------------------------------------- #
# Stop conditions
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class StopConditions:
    """When to stop scheduling. All ``None``/default ⇒ run forever.

    ``max_consecutive_failures`` is a circuit-breaker for a *capital-preservation*
    monitor: a tick that returns ``failed=True`` OR raises (isolated by the loop)
    increments a running counter; a non-failing tick resets it. When the counter
    reaches the threshold the loop stops with a **non-zero** return code (a clean
    deadline / max-activations / agent-signal stop still returns 0). ``None``
    (default) ⇒ never trip — keep watching through any number of failures.

    ``stop_on_signal`` controls only the *agent in-band* stop channel
    (``[[SCHEDULE_STOP]]`` / ``ccx_schedule_stop`` tag), not OS signals.
    """

    max_activations: int | None = None
    until: datetime | None = None
    stop_on_signal: bool = True
    max_consecutive_failures: int | None = None

    def __post_init__(self) -> None:
        if self.max_activations is not None and self.max_activations < 1:
            raise ValueError("StopConditions: max_activations must be >= 1 (or None)")
        if (
            self.max_consecutive_failures is not None
            and self.max_consecutive_failures < 1
        ):
            raise ValueError(
                "StopConditions: max_consecutive_failures must be >= 1 (or None)"
            )


# --------------------------------------------------------------------------- #
# Interruptible sleep
# --------------------------------------------------------------------------- #

def _interruptible_sleep_until(
    wake_at: datetime,
    sleeper: Callable[[float], None],
    wall_clock: Callable[[], datetime],
    *,
    slice_s: float = 1.0,
) -> None:
    """Sleep in bounded slices until ``wake_at``, re-checking the clock.

    Bounded slices keep SIGINT/SIGTERM latency to ~``slice_s`` even for long
    ``daily`` waits. A ``KeyboardInterrupt`` raised by ``sleeper`` propagates
    (the loop converts it to a clean exit).
    """
    while True:
        now = wall_clock()
        remaining = (wake_at - now).total_seconds()
        if remaining <= 0:
            return
        sleeper(min(slice_s, remaining))


# --------------------------------------------------------------------------- #
# OS signal handling (daemon surface)
# --------------------------------------------------------------------------- #

def _raise_keyboard_interrupt(_signum: int, _frame: Any) -> None:
    """Turn an OS termination signal into the loop's clean-exit path."""
    raise KeyboardInterrupt


@contextmanager
def _install_signal_handlers(
    signums: Sequence[int] = (signal.SIGINT, signal.SIGTERM),
) -> Iterator[None]:
    """Route SIGINT/SIGTERM into ``KeyboardInterrupt`` for a clean rc-0 exit.

    Installed only by the CLI/daemon surface (``main``): the run loop catches
    ``KeyboardInterrupt`` and returns 0, so both signals stop the schedule
    gracefully — including a daemon launched in the background, where SIGINT is
    otherwise inherited as ``SIG_IGN`` and SIGTERM would hard-kill (rc 143).
    Prior handlers are restored on exit. A no-op (handlers left untouched) when
    not on the main thread — ``signal.signal`` raises there — so the Python API
    and tests never have their host's signal disposition rewritten.
    """
    previous: dict[int, Any] = {}
    try:
        for sig in signums:
            try:
                previous[sig] = signal.signal(sig, _raise_keyboard_interrupt)
            except (ValueError, OSError, RuntimeError):
                # Not the main thread / unsupported signal on this platform.
                pass
        yield
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError, RuntimeError):
                pass


# --------------------------------------------------------------------------- #
# Per-tick request construction + stop detection
# --------------------------------------------------------------------------- #

def _build_tick_request(
    template: Any,
    previous_run_id: str | None,
    activation: int,
    bridge: str,
) -> Any:
    """Stamp per-tick metadata onto a copy of the template (never mutate it).

    State bridges to the next tick via the existing resume metadata
    (``bridge in {'resume','both'}``) and cross-run memory (always: a
    ``monitoring`` tag is added so recall/summaries stay scoped).
    """
    meta = dict(getattr(template, "metadata", None) or {})
    if bridge in ("resume", "both") and previous_run_id:
        meta[RESUME_PREVIOUS_RUN_METADATA_KEY] = previous_run_id
    meta[SCHEDULE_METADATA_KEY] = {
        "activation": activation,
        "previous_run_id": previous_run_id,
    }
    tags = _merge_memory_tags(meta.get("ccx_memory_tags"))
    meta["ccx_memory_tags"] = list(tags)
    return replace(template, metadata=meta)


def _merge_memory_tags(existing: Any) -> tuple[str, ...]:
    # MONITORING_TAG goes FIRST: the memory write/recall paths re-run
    # ``normalize_tags`` which hard-caps at MAX_TAGS (10), keeping the leading
    # tags. With many caller-supplied tags, appending ``monitoring`` last would
    # let the truncation drop the one tag that scopes the whole monitoring
    # history — so it must lead.
    out: list[str] = [MONITORING_TAG]
    if isinstance(existing, str):
        out.append(existing)
    elif isinstance(existing, (list, tuple)):
        out.extend(str(t) for t in existing)
    # de-dup, order-preserving (monitoring stays at index 0)
    seen: set[str] = set()
    deduped: list[str] = []
    for t in out:
        if t and t not in seen:
            seen.add(t)
            deduped.append(t)
    return tuple(deduped)


def _result_run_id(result: Any) -> str:
    return str(getattr(result, "session_id", "") or "")


def _agent_requested_stop(result: Any, cwd: str | None) -> bool:
    """True when the agent signalled "stop the schedule" for this tick.

    Two channels, checked cheapest-first:
    1. ``[[SCHEDULE_STOP]]`` in the tick's ``final_text`` — the **robust**
       channel: read straight off the result, never persisted, so never subject
       to memory de-duplication. Prefer it.
    2. A memory entry tagged ``ccx_schedule_stop`` whose ``run_id`` is this
       tick's run. Durable, but **best-effort**: the memory store de-duplicates
       on ``(kind, title, text)`` — NOT ``run_id`` — so if the agent writes a
       byte-identical stop note that some earlier run already wrote (e.g. after
       a supervisor restart re-using the same note), the newer entry is dropped
       and this run-scoped lookup will miss it. Keep stop notes run-unique
       (embed the run_id / a timestamp) or rely on channel 1. Run-scoping is
       intentional so a stale stop tag from a prior campaign can't halt a fresh
       schedule.
    """
    final_text = str(getattr(result, "final_text", "") or "")
    if SCHEDULE_STOP_SENTINEL in final_text:
        return True

    run_id = _result_run_id(result)
    if not run_id:
        return False
    try:
        from .memory.store import JsonlMemoryStore

        root = _memory_root_for(cwd)
        store = JsonlMemoryStore(root)
        entries, _ = store.load()
        for entry in entries:
            if entry.run_id == run_id and SCHEDULE_STOP_TAG in (entry.tags or ()):
                return True
    except Exception:  # noqa: BLE001 — stop-detection must never crash the loop
        logger.debug("schedule: stop-tag check failed", exc_info=True)
    return False


def _memory_root_for(cwd: str | None) -> Path:
    """Default memory root (mirrors api._memory_root's default branch)."""
    try:
        base = Path(cwd or ".").resolve()
    except (OSError, ValueError):
        base = Path(cwd or ".")
    return base / ".ccx" / "memory"


# --------------------------------------------------------------------------- #
# The supervisor loop
# --------------------------------------------------------------------------- #

def run_schedule(
    *,
    agent: Any,
    request_template: Any,
    schedule: ScheduleSpec,
    stop: StopConditions,
    bridge: str = "memory",
    sink: Any = None,
    fire_immediately: bool = False,
    sleeper: Callable[[float], None] = time.sleep,
    wall_clock: Callable[[], datetime] = _now_local,
) -> int:
    """Drive a fresh bounded ``agent.run_sync`` per scheduled tick.

    Returns 0 on a clean stop (deadline / max activations / agent signal /
    ``KeyboardInterrupt``) and a **non-zero** code only when the
    ``max_consecutive_failures`` circuit-breaker trips (monitoring aborted on
    repeated failure). ``agent`` need only expose ``run_sync(request)`` returning
    an object with ``session_id`` / ``final_text`` / ``failed``.

    A single tick that *raises* (e.g. a fail-loud ``CancelledError`` or a
    teardown hiccup escaping ``run_sync``) is isolated: the loop emits a
    ``tick_error`` event and proceeds to the next tick rather than letting one
    bad tick take the whole monitor dark. ``KeyboardInterrupt`` / ``SystemExit``
    are never swallowed — they remain the clean-shutdown path.

    ``bridge`` selects the state-bridge between ticks: ``memory`` (default;
    distilled tag-scoped history), ``resume`` (detailed prior-run snapshot via
    resume metadata), or ``both``. ``fire_immediately`` makes the *first* tick
    fire at once (used by ``--once`` smoke tests); the steady-state cadence is
    unchanged.
    """
    if bridge not in ("memory", "resume", "both"):
        raise ValueError(f"run_schedule: unknown bridge {bridge!r}")

    cwd = getattr(request_template, "cwd", None)
    previous_run_id: str | None = None
    last_fire: datetime | None = None
    activation = 0
    consecutive_failures = 0

    try:
        while True:
            now = wall_clock()
            if stop.until is not None and now >= stop.until:
                _emit(sink, "info", "stop", {"reason": "deadline", "activation": activation})
                return 0
            if fire_immediately and activation == 0:
                wake_at = now
            else:
                wake_at = schedule.next_fire_at(now, last_fire)
            # Inclusive boundary: "no new tick starts at/after the deadline"
            # (matches the top-of-loop ``now >= until`` gate).
            if stop.until is not None and wake_at >= stop.until:
                _emit(sink, "info", "stop", {"reason": "deadline", "activation": activation})
                return 0

            _interruptible_sleep_until(wake_at, sleeper, wall_clock)
            last_fire = wake_at
            activation += 1

            request = _build_tick_request(
                request_template, previous_run_id, activation, bridge,
            )
            _emit(sink, "info", "tick_start", {
                "activation": activation,
                "wake_at": wake_at.isoformat(),
                "previous_run_id": previous_run_id,
            })

            # Per-tick exception isolation. ``run_sync`` returns ``failed=True``
            # for most operational errors, but a few paths still raise out of it
            # (fail-loud ``CancelledError``, ``_build_result`` / ``bundle.shutdown``
            # / memory-finalize after the engine catch, asyncio.run setup). A
            # monitoring loop must survive one bad tick, so isolate it and treat
            # it like a failed tick. KeyboardInterrupt/SystemExit propagate.
            result: Any = None
            try:
                result = agent.run_sync(request)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:  # noqa: BLE001 — incl. CancelledError
                tick_failed = True
                logger.warning(
                    "schedule: tick %d raised %s; isolating and continuing",
                    activation, type(exc).__name__, exc_info=True,
                )
                _emit(sink, "warn", "tick_error", {
                    "activation": activation,
                    "error": type(exc).__name__,
                    "message": str(exc),
                })
            else:
                run_id = _result_run_id(result)
                if run_id:
                    previous_run_id = run_id
                tick_failed = bool(getattr(result, "failed", False))
                _emit(sink, "warn" if tick_failed else "info", "tick_done", {
                    "activation": activation,
                    "run_id": run_id,
                    "failed": tick_failed,
                })

            consecutive_failures = consecutive_failures + 1 if tick_failed else 0

            # Stop checks: agent in-band signal (only meaningful with a result),
            # then the failure circuit-breaker (abnormal abort), then the planned
            # max-activations completion.
            if (
                result is not None
                and stop.stop_on_signal
                and _agent_requested_stop(result, cwd)
            ):
                _emit(sink, "info", "stop", {
                    "reason": "agent_signal", "activation": activation,
                })
                return 0
            if (
                stop.max_consecutive_failures is not None
                and consecutive_failures >= stop.max_consecutive_failures
            ):
                _emit(sink, "warn", "stop", {
                    "reason": "max_consecutive_failures",
                    "activation": activation,
                    "consecutive_failures": consecutive_failures,
                })
                return 1
            if (
                stop.max_activations is not None
                and activation >= stop.max_activations
            ):
                _emit(sink, "info", "stop", {
                    "reason": "max_activations", "activation": activation,
                })
                return 0
    except KeyboardInterrupt:
        _emit(sink, "info", "stop", {"reason": "interrupt", "activation": activation})
        return 0


def _emit(sink: Any, severity: str, source: str, payload: dict[str, Any]) -> None:
    if sink is None:
        return
    try:
        sink.emit(severity, source, payload)
    except Exception:  # noqa: BLE001 — logging must never break the loop
        logger.debug("schedule: sink.emit failed", exc_info=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m core.ccx.scheduler")
    p.add_argument("--cwd", default=".", help="ccx workspace for the agent runs")
    p.add_argument("--goal", required=True, help="the instruction run each tick")
    p.add_argument(
        "--agent-mode", default="agent",
        help="ccx agent_mode for each tick (default 'agent')",
    )
    trig = p.add_mutually_exclusive_group(required=True)
    trig.add_argument(
        "--interval", type=float, default=None,
        help="fire every N seconds",
    )
    trig.add_argument(
        "--at", action="append", default=None, metavar="HH:MM",
        help="fire daily at this local time (repeatable)",
    )
    p.add_argument(
        "--weekdays", default=None,
        help="restrict --at to weekdays, e.g. 'mon-fri' or 'mon,wed,fri' (daily only)",
    )
    p.add_argument(
        "--catch-up", action="store_true",
        help="interval: anchor cadence to last fire so overruns fire immediately",
    )
    p.add_argument(
        "--bridge", choices=("memory", "resume", "both"), default="memory",
        help="state bridge between ticks (default 'memory')",
    )
    p.add_argument("--max-activations", type=int, default=None, help="stop after N ticks")
    p.add_argument(
        "--until", default=None,
        help=(
            "ISO datetime; no new tick starts at/after this deadline. A bare "
            "date (YYYY-MM-DD) means through the end of that local day. A past "
            "deadline is rejected."
        ),
    )
    p.add_argument(
        "--once", action="store_true",
        help="run a single tick and exit (smoke test; implies max-activations=1)",
    )
    p.add_argument("--log-file", default=None, help="append JSONL tick records here")
    p.add_argument("--quiet", action="store_true", help="suppress severity=info on stderr")
    return p


def _schedule_from_args(args: argparse.Namespace) -> ScheduleSpec:
    if args.interval is not None:
        return ScheduleSpec(
            kind="interval",
            interval_seconds=float(args.interval),
            catch_up=bool(args.catch_up),
        )
    return ScheduleSpec(
        kind="daily",
        daily_times=tuple(args.at or ()),
        weekdays=_parse_weekdays(args.weekdays),
    )


def _validate_cli_combo(args: argparse.Namespace) -> str | None:
    """Reject silently-ineffective flag combinations (returns an error message).

    argparse only makes --interval/--at mutually exclusive; the modifier flags
    bind to one kind, so the wrong pairing would otherwise be dropped without a
    word (e.g. ``--interval 300 --weekdays mon-fri`` runs unrestricted 24/7).
    """
    if args.interval is not None and args.weekdays is not None:
        return "--weekdays only applies to --at (daily) schedules"
    if args.at and args.catch_up:
        return "--catch-up only applies to --interval schedules"
    if args.max_activations is not None and args.max_activations < 1:
        return "--max-activations must be >= 1"
    return None


def _parse_until(raw: str, now: datetime) -> tuple[datetime | None, str | None]:
    """Parse ``--until`` to an aware deadline (or an error message).

    A bare date (e.g. ``2026-12-31``) is interpreted as *inclusive of that whole
    local day* — the deadline is the following local midnight — so a date-only
    deadline does not stop the schedule a day early. A past deadline is a loud
    operator error (it would silently run nothing).
    """
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        return None, f"invalid --until {raw!r}: {exc}"
    if dt.tzinfo is None:
        dt = dt.astimezone()
    date_only = ("T" not in raw) and (":" not in raw)
    if date_only:
        dt = dt + timedelta(days=1)  # through the end of that calendar day
    if dt <= now:
        return None, f"--until {raw!r} is in the past; nothing would run"
    return dt, None


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not schedule_enabled():
        print(
            "error: scheduling is gated off. Set CCX_SCHEDULE_ENABLE=1 to run "
            "the scheduler daemon (use --once with the flag for a smoke test).",
            file=sys.stderr,
        )
        return 2

    combo_error = _validate_cli_combo(args)
    if combo_error is not None:
        print(f"error: {combo_error}", file=sys.stderr)
        return 2

    try:
        schedule = _schedule_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    until: datetime | None = None
    if args.until:
        until, until_error = _parse_until(args.until, _now_local())
        if until_error is not None:
            print(f"error: {until_error}", file=sys.stderr)
            return 2

    max_activations = args.max_activations
    if args.once:
        max_activations = 1

    try:
        stop = StopConditions(max_activations=max_activations, until=until)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Lazy import so importing this module (and unit-testing the pure pieces)
    # never pulls the full agent/LLM stack.
    from core.cc.api import AgentRunRequest
    from .api import CodeAgent
    from .llm_monitor import AlertSink

    agent = CodeAgent(agent_runner_kind="auto")
    request_template = AgentRunRequest(
        instruction=args.goal,
        cwd=args.cwd,
        agent_mode=args.agent_mode,
    )
    sink = AlertSink(
        log_file=Path(args.log_file) if args.log_file else None,
        quiet=args.quiet,
    )

    # Daemon surface: convert SIGINT+SIGTERM into the clean rc-0 exit path so a
    # backgrounded/orchestrated supervisor shuts down gracefully instead of
    # hard-killing (SIGTERM rc 143) or ignoring Ctrl-C (inherited SIG_IGN).
    with _install_signal_handlers():
        return run_schedule(
            agent=agent,
            request_template=request_template,
            schedule=schedule,
            stop=stop,
            bridge=args.bridge,
            sink=sink,
            fire_immediately=bool(args.once),
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ScheduleSpec",
    "StopConditions",
    "run_schedule",
    "schedule_enabled",
    "SCHEDULE_STOP_TAG",
    "SCHEDULE_STOP_SENTINEL",
    "MONITORING_TAG",
    "SCHEDULE_METADATA_KEY",
    "main",
]
