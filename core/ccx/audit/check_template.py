"""Hermetic-safety validator for synthesized ``[check:]`` commands.

Shared contract between regression-capture (which *scaffolds* a repro check) and
the debug binding (which *runs* it inside the goal loop). A check is the trust
root, and it runs under the hermetic env (see :mod:`core.ccx.sgar.checks`):
``PYTHONSAFEPATH=1`` strips the cwd from ``sys.path`` and ``PYTHONPATH`` is
dropped. So a cwd-relative ``python -c "import workspace_mod"`` or
``python workspace_script.py`` would FAIL hermetically — and that failure is
*mechanically indistinguishable* from a real RED, silently dooming the loop.

The checks.py docstring's prescribed safe form is ``pytest <testfile>`` (pytest
inserts the rootdir into ``sys.path`` itself, so it passes hermetically). For
the checks WE synthesize we therefore enforce exactly that allowlist: a check
must be a pytest invocation. This is intentionally stricter than "not obviously
unsafe" — the code-task contract's ``python -c`` bootstrap (which pins an
ABSOLUTE root and is hermetic-safe) is built elsewhere and never passes through
this validator.
"""

from __future__ import annotations

import shlex
from pathlib import Path

__all__ = ["is_pytest_check", "validate_check_command"]


def is_pytest_check(command: str) -> bool:
    """True iff ``command`` is a pytest invocation (``pytest …`` / ``python -m
    pytest …``) — the hermetic-safe form for a synthesized check."""
    try:
        parts = shlex.split(command or "")
    except ValueError:
        return False
    if not parts:
        return False
    head = Path(parts[0]).name
    if head == "pytest":
        return True
    if head.startswith("python"):
        if "-m" in parts:
            i = parts.index("-m")
            return i + 1 < len(parts) and parts[i + 1] == "pytest"
    return False


def validate_check_command(command: str) -> None:
    """Raise ``ValueError`` unless ``command`` is a hermetic-safe pytest check.

    Use at the boundary where a repro check is accepted/scaffolded so a
    cwd-relative ``python -c import …`` / ``python script.py`` can never become a
    silently-doomed loop check.
    """
    if not is_pytest_check(command):
        raise ValueError(
            f"check command must be a pytest invocation "
            f"('pytest <file>' or 'python -m pytest <file>'); got {command!r}. "
            "A cwd-relative `python -c \"import ...\"` or `python script.py` "
            "would FAIL under the hermetic check env (PYTHONSAFEPATH strips cwd "
            "from sys.path) and be indistinguishable from a real RED."
        )
