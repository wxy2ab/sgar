"""Execution of machine-checkable exit-criterion commands.

A stage spec may annotate an exit criterion with a ``[check: <command>]``
suffix (see :func:`core.ccx.sgar.validation.parse_exit_criteria`). When the
runtime has check execution enabled (opt-in, default off), SGAR runs that
command itself during verify/close instead of trusting the agent's
self-reported pass/fail. This module is the deterministic executor:

* exit code 0 → check passed; non-zero / timeout / spawn error → failed.
* the command runs through ``shlex.split`` (NO shell), so a spec that needs a
  pipeline must wrap it explicitly and visibly (``sh -c "..."``).
* a bounded tail of combined stdout/stderr is captured as machine evidence so
  the verification report and trace show *what the machine saw*, not prose.

The runtime is responsible for the opt-in gate and for the cwd / timeout it
passes in; this module never decides whether a check *should* run.

Hermetic execution — the trust root
-----------------------------------
A ``[check:]`` exit code is the trust root of the whole governance stack: every
gate, ``goal``/``sgar``/``sgarx`` verdict, and run-level audit ultimately rests
on the number this module observes. A capable agent with file tools can subvert
that number *without legitimately satisfying the criterion* by poisoning the
Python interpreter the check spawns — this was demonstrated empirically (PR#22
reasoning honesty matrix): an agent planted a ``sitecustomize`` in the user-site
(``~/.local/.../site-packages``, which escapes the workspace and *globally*
re-patches ``sys.exit``) to turn an impossible gate's ``sys.exit(7)`` into ``0``.

So by default we run each check in a *hermetic* environment that strips the
Python interpreter-poisoning vectors while preserving the real toolchain
(``PATH`` and the active venv):

* ``PYTHONNOUSERSITE=1`` — no user-site, so no user-site ``usercustomize`` /
  ``sitecustomize`` startup hook (the demonstrated vector).
* ``PYTHONSAFEPATH=1`` (CPython ≥ 3.11; inert below) — don't auto-prepend the
  cwd / script dir to ``sys.path``, so the check can't ``import`` a
  workspace-relative module an agent dropped next to it.
* drop ``PYTHONPATH`` / ``PYTHONSTARTUP`` / ``PYTHONHOME`` — no path-injected
  module, no startup hook, no relocated stdlib.

These flags do not change the *toolchain* a check resolves (``pytest``,
``python -m pytest``, ``ruff`` all come from ``PATH`` / the venv exactly as
before); a check that passes normally but FAILS hermetically is a
gate-subversion signal, not a flaky re-run. ``CCX_HERMETIC_CHECKS=0`` restores
the previous (inherit-parent-env) behavior byte-for-byte as an escape hatch.

One legitimate casualty, *by design*: a check that ``import``s a module the
agent just wrote into its **workspace** — e.g. ``python -c "from new_mod import
f; assert f(...)"`` run with ``cwd`` = workspace — is also blocked, because
``PYTHONSAFEPATH`` strips cwd from ``sys.path`` and a legitimate workspace import
is *mechanically indistinguishable* from the poison drop above. We accept that
rather than reopen the vector. Write such a check as ``pytest test_x.py``
(pytest inserts the rootdir into ``sys.path`` itself, so it still passes
hermetically) — NOT ``python -c "import new_mod"`` or ``python ws_script.py``
that lean on the stripped cwd / script-dir — or set ``CCX_HERMETIC_CHECKS=0``
for that run.

THREAT BOUNDARY — do not over-trust this.
Hermetic checks blunt the *cheapest, already-demonstrated* vector: Python
interpreter poisoning. They are NOT adversarial containment. An agent with
file/exec tools can still subvert a check by other means — writing the exact
artifact the check looks for, shadowing a binary earlier on ``PATH``, editing
the repo under test, or having the check shell out to a non-Python program it
controls. Real containment against an agent that *deliberately* subverts its
execution environment requires an OS-level sandbox (namespaces / seccomp /
containers), which is out of scope for ccx. ccx's governance model assumes an
agent that may err or lie in its *self-report* but does not actively poison its
runtime; this hardening narrows — it does not close — the gap to that
assumption.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import ExitCriterion, SgarError


_EVIDENCE_TAIL_LINES = 20
_EVIDENCE_MAX_CHARS = 2000

# Env hardening for machine checks (see module docstring). On by default; set
# ``CCX_HERMETIC_CHECKS`` to a falsey value to inherit the parent env unchanged.
_HERMETIC_FLAG_ENV = "CCX_HERMETIC_CHECKS"
_HERMETIC_FALSEY = frozenset({"0", "false", "off", "no", ""})
_HERMETIC_DROP_VARS = ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME")

# Opt-in (default OFF) early-stop when a governed loop's failing checks are ALL
# unrunnable (harness defects). See ``stop_on_unrunnable_enabled`` and the three
# governed loops' no-progress handling.
_STOP_ON_UNRUNNABLE_ENV = "CCX_STOP_ON_UNRUNNABLE"
_TRUTHY = frozenset({"1", "true", "on", "yes"})


def stop_on_unrunnable_enabled() -> bool:
    """Whether a governed loop should stop EARLY when every failing check this
    round is unrunnable (a harness defect a re-drive can never repair).

    Default OFF — ``CCX_STOP_ON_UNRUNNABLE`` unset (or set to anything other than
    ``1``/``true``/``on``/``yes``) keeps the legacy control flow byte-for-byte:
    the loop logs the harness-defect warning but otherwise falls through to the
    normal no-progress / max-iters bound. Opt in to convert "all failures are
    unrunnable" into an immediate ``stop_reason="harness_defect"`` so the loop
    doesn't burn its iteration budget re-running a check that cannot execute.
    """
    raw = os.environ.get(_STOP_ON_UNRUNNABLE_ENV)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY


def _hermetic_checks_enabled() -> bool:
    """Whether to spawn machine checks in a sanitized environment (default ON).

    ``CCX_HERMETIC_CHECKS=0`` (or ``false``/``off``/``no``/empty) opts out and
    inherits the parent environment unchanged — a byte-equivalent escape hatch
    for the rare legitimate check that needs a user-site package or a
    ``PYTHONPATH`` entry.
    """
    raw = os.environ.get(_HERMETIC_FLAG_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in _HERMETIC_FALSEY


def _hermetic_env() -> dict[str, str] | None:
    """Sanitized copy of ``os.environ`` for spawning a check, or ``None`` to
    inherit the parent environment unchanged (escape hatch / opt-out).

    Strips the Python interpreter-poisoning vectors (user-site startup hooks,
    ``PYTHONPATH``/``PYTHONSTARTUP``/``PYTHONHOME`` injection, cwd-on-path)
    while *preserving* ``PATH`` and the active venv so ordinary checks are
    unaffected. See the module docstring for the threat boundary: this narrows,
    it does not contain.
    """
    if not _hermetic_checks_enabled():
        return None
    env = dict(os.environ)
    for key in _HERMETIC_DROP_VARS:
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONSAFEPATH"] = "1"  # honored on CPython >= 3.11; inert otherwise
    return env


@dataclass(slots=True)
class CheckOutcome:
    """Result of running one criterion's machine check."""

    criterion_id: str
    command: str
    passed: bool
    returncode: int | None  # None on timeout / spawn failure
    output_tail: str = ""
    timed_out: bool = False
    error: str | None = None  # spawn-time error (bad command, not found, …)

    def evidence_line(self) -> str:
        """One-line (+ optional output tail) machine-evidence summary."""
        if self.timed_out:
            status = "TIMEOUT"
        elif self.error is not None:
            status = f"ERROR: {self.error}"
        else:
            status = f"exit={self.returncode}"
        verdict = "PASS" if self.passed else "FAIL"
        head = f"machine check `{self.command}` -> {verdict} ({status})"
        if self.output_tail:
            return f"{head}\n{self.output_tail}"
        return head


def check_unrunnable(outcome: CheckOutcome) -> bool:
    """Best-effort: did this check FAIL TO EXECUTE (vs run and report false)?

    A check that cannot run (malformed command, missing binary, shell syntax
    error, timeout) provides NO verification signal. Because the verification
    spec is immutable across iterations, such a check can never be repaired —
    it silently dooms the run with a *false* not-met and triggers pointless
    re-drives. We flag it so the operator can tell a plan/harness defect apart
    from a genuine condition failure. Detection is conservative: a check that
    *ran* and exited non-zero for a real reason (e.g. diff mismatch, rc=1) is
    NOT flagged — only unambiguous non-execution.

    Shared by all three governed loops (``governed_goal`` / ``governed_run`` /
    ``governed_spawn``) so the unrunnable-vs-genuine-failure distinction is the
    SAME predicate everywhere; it lives here, next to ``CheckOutcome``, rather
    than in any one loop.
    """
    if outcome.timed_out:
        return True
    rc = outcome.returncode
    if rc is None:            # spawn failure / FileNotFoundError / timeout
        return True
    if rc == 127:             # command not found (via `sh -c`)
        return True
    if rc == 2 and "syntax error" in (outcome.output_tail or "").lower():
        return True            # shell could not parse the command at all
    return False


def _tail(text: str) -> str:
    if not text:
        return ""
    lines = text.strip().splitlines()
    tail = "\n".join(lines[-_EVIDENCE_TAIL_LINES:])
    if len(tail) > _EVIDENCE_MAX_CHARS:
        tail = tail[-_EVIDENCE_MAX_CHARS:]
    return tail


def _coerce_stream(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run_criterion_check(
    criterion: ExitCriterion,
    *,
    cwd: str | Path,
    timeout_s: float,
) -> CheckOutcome:
    """Run a criterion's machine check and report the outcome.

    Never raises for an ordinary failed/timed-out/unspawnable command — those
    are reported as ``passed=False`` outcomes so the caller can attach
    evidence and decide policy. Only raises :class:`SgarError` for the
    programming error of calling this on a criterion with no check.
    """
    command = (criterion.check or "").strip()
    if not command:
        raise SgarError(
            f"criterion {criterion.criterion_id} has no machine check to run"
        )
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return CheckOutcome(
            criterion_id=criterion.criterion_id,
            command=command,
            passed=False,
            returncode=None,
            error=f"could not parse check command: {exc}",
        )
    if not argv:
        return CheckOutcome(
            criterion_id=criterion.criterion_id,
            command=command,
            passed=False,
            returncode=None,
            error="empty check command",
        )
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_hermetic_env(),
        )
    except subprocess.TimeoutExpired as exc:
        tail = _tail(_coerce_stream(exc.stdout) + _coerce_stream(exc.stderr))
        return CheckOutcome(
            criterion_id=criterion.criterion_id,
            command=command,
            passed=False,
            returncode=None,
            output_tail=tail,
            timed_out=True,
        )
    except (OSError, ValueError) as exc:
        return CheckOutcome(
            criterion_id=criterion.criterion_id,
            command=command,
            passed=False,
            returncode=None,
            error=str(exc),
        )
    tail = _tail(_coerce_stream(proc.stdout) + _coerce_stream(proc.stderr))
    return CheckOutcome(
        criterion_id=criterion.criterion_id,
        command=command,
        passed=proc.returncode == 0,
        returncode=proc.returncode,
        output_tail=tail,
    )


__all__ = [
    "CheckOutcome",
    "check_unrunnable",
    "run_criterion_check",
    "stop_on_unrunnable_enabled",
]
