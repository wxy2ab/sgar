"""Version-bound audit detail for candidate-conditioned repair feeds.

中文：接口优势是 **C（candidate-conditioned audit）**，不是 B（从零构造）。
回喂文本必须绑定 ``candidate_hash``，禁止把旧 residual 用在新候选上。

English: Prefer candidate-conditioned audit (interface C) over from-scratch
construction (B). Feed-back text must bind ``candidate_hash`` so old residuals
are never applied to a new candidate.
"""

from __future__ import annotations

from .models import CandidateRecord


def format_candidate_bound_detail(
    candidate: CandidateRecord,
    *,
    findings: list[str],
    prev_candidate_hash: str | None = None,
    include_patch_hint: bool = False,
) -> str:
    """Build a VERSION-BOUND AUDIT block aligned with syndrome / autobuild style."""
    lines = [
        "=== VERSION-BOUND AUDIT ===",
        f"candidate_id: {candidate.candidate_id}",
        f"candidate_hash: {candidate.candidate_hash}",
        f"git_head: {candidate.git_head or '-'}",
        f"status: {candidate.status}",
        "---",
    ]
    if findings:
        lines.extend(str(item) for item in findings)
    else:
        lines.append("(no findings)")
    lines.append("---")
    prev = str(prev_candidate_hash or "").strip()
    if prev and prev != candidate.candidate_hash:
        lines.append(
            f"NOTE: evidence may belong to a PREVIOUS candidate "
            f"(was {prev}); re-read named paths before acting."
        )
    if include_patch_hint:
        from ...agents.patch_first import patch_first_hint

        lines.append(f"PATCH-FIRST: {patch_first_hint()}")
    return "\n".join(lines)


__all__ = ["format_candidate_bound_detail"]
