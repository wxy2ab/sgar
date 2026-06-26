"""Stage-Governed Agent Runtime for ccx."""

from .autobuild import (
    AutobuildReport,
    ProjectPlan,
    StagePlan,
    StageReport,
    autobuild,
)
from .checks import CheckOutcome, run_criterion_check
from .models import (
    CriterionResult,
    DoctorResult,
    ExitCriterion,
    ProjectMode,
    ProjectState,
    SgarError,
    StageRecord,
    StageStatus,
    ValidationResult,
    VerificationReport,
)
from .runtime import SgarRuntime, result_to_text
from .store import SgarStore
from .missions import mission_staleness
from .tracing import TRACE_FILENAME, read_failed_trace, read_trace

__all__ = [
    "AutobuildReport",
    "CheckOutcome",
    "CriterionResult",
    "DoctorResult",
    "ExitCriterion",
    "ProjectMode",
    "ProjectPlan",
    "ProjectState",
    "SgarError",
    "SgarRuntime",
    "SgarStore",
    "StagePlan",
    "StageRecord",
    "StageReport",
    "StageStatus",
    "TRACE_FILENAME",
    "ValidationResult",
    "VerificationReport",
    "autobuild",
    "mission_staleness",
    "read_failed_trace",
    "read_trace",
    "result_to_text",
    "run_criterion_check",
]
