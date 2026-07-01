"""DeliberateRunner — adversarial proposal deliberation (Phase 0, single-round).

Design: ``docs/deliberate_mode_design_2026-06-30.md``. Pipeline:

1. **decompose** — one bounded LLM turn splits the proposal into ≤ ``max_subclaims``
   independently-investigable sub-claims;
2. **adversarial research** — per sub-claim, two stance-committed read-only
   :class:`ResearchRunner` workers (``support`` / ``refute``) run in parallel and
   return evidence-anchored findings;
3. **arbitrate** — a *deterministic* (no-LLM) cite-or-discount step computes a
   three-state verdict (``leans-for`` / ``leans-against`` / ``uncertain``) from
   each side's count of **grounded evidence anchors** (file:line), never from
   rhetorical strength;
4. **aggregate** — roll the sub-claim verdicts into an overall for/against view.

Phase 0 deliberately makes the **verdict deterministic Python** (steps 3-4): the
LLM only *decomposes* and *gathers* evidence (where its strength lies); Python
*disposes* by tallying grounded anchors. This is the deliberation analogue of the
audit doctrine ("LLM proposes, Python disposes") and makes the four design
disciplines (cite-or-discount, honest-uncertain, bounded, stance-independent)
real mechanisms — fully testable without an LLM. An LLM narrative-synthesis layer
and bounded adaptive depth are Phase 1 (do not add until single-round earns it).

INFORM-only by construction: the result dict is attached to
``snapshot['deliberation']`` and is never a term in any gate.

Bounding (the 7h-hang lesson): the decompose LLM turn routes through
``_call_llm_bounded`` (daemon thread + ``join(timeout)``); the two stance workers
run on daemon threads joined with ``research_timeout_s``. Never
``ThreadPoolExecutor`` (its atexit join can deadlock on a hung stream).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from core.cc.config import CCConfig

from .research_runner import ResearchRunner
from .subagent import SubagentInvocation
from ..modes.llm_client import LLMCallable, text_of
from ..modes.parsing import parse_llm_json

logger = logging.getLogger(__name__)


def _bounded_llm_text(
    llm: LLMCallable, *, system: str, user: str, purpose: str, timeout_s: float,
) -> str:
    """Call ``llm`` on a daemon thread joined with ``timeout_s``; "" on timeout/error.

    Self-contained bounded wrapper (the same daemon-thread + ``join(timeout)``
    pattern as ``governed_goal._call_llm_bounded`` / ``audit_runner``), kept local
    so this module does not import a private cross-module helper. A daemon thread
    (never ``ThreadPoolExecutor``) is deliberate: a hung reasoning client is
    abandoned at process exit rather than deadlocking ``atexit`` (the 7h-hang
    vector). An empty string is treated by ``parse_llm_json`` as a parse failure →
    the caller's safe fallback.
    """
    box: dict[str, Any] = {"text": "", "done": False}

    def _worker() -> None:
        try:
            box["text"] = text_of(llm(system=system, user=user, purpose=purpose))
        except Exception:  # noqa: BLE001 — never let a worker crash the run
            logger.warning(
                "ccx deliberate: LLM call %s raised", purpose, exc_info=True,
            )
        finally:
            box["done"] = True

    thread = threading.Thread(
        target=_worker, name=f"ccx-deliberate-{purpose}", daemon=True,
    )
    thread.start()
    thread.join(timeout_s)
    if not box["done"]:
        logger.warning(
            "ccx deliberate: LLM call %s did not return within %.0fs; "
            "abandoning the thread and falling back", purpose, timeout_s,
        )
        return ""
    return str(box["text"] or "")

__all__ = [
    "DeliberateRunner",
    "count_grounded_anchors",
    "arbitrate",
    "aggregate",
    "render_deliberation_report",
]


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_DECOMPOSE_SYSTEM_PROMPT = """\
You are a deliberation planner. You are given a PROPOSAL (a proposition someone \
is considering — NOT a task to execute). Split it into a small set of \
INDEPENDENT, investigable SUB-CLAIMS: the load-bearing factual assertions the \
proposal stands or falls on, each phrasable as something a read-only code \
investigation could find evidence for OR against.

Rules:
  * Each sub-claim is ONE assertion, stated so evidence could support or refute \
it (avoid vague value judgements).
  * Prefer the few claims that actually decide the proposal; do not pad.
  * Do NOT take a side. Do NOT answer them. Just name them.

Output a final message with THIS exact JSON shape and no surrounding prose:

{"subclaims": ["<claim 1>", "<claim 2>", ...]}
"""


def _build_decompose_user_prompt(proposal: str, max_subclaims: int) -> str:
    return (
        f"## PROPOSAL\n{(proposal or '').strip() or '(empty proposal)'}\n\n"
        f"Emit at most {max_subclaims} sub-claims as JSON now."
    )


#: Stance directives injected into the read-only research worker. The worker is
#: still bound by ResearchRunner's "cite only real code, do not fabricate" rules;
#: the stance only steers WHICH direction it hunts for grounded evidence.
_RESEARCH_QUESTION = (
    "Evaluate this claim against the codebase and report grounded evidence: {claim}"
)


# --------------------------------------------------------------------------- #
# Deterministic arbitration (the cite-or-discount "teeth")
# --------------------------------------------------------------------------- #

def count_grounded_anchors(finding: dict[str, Any] | None) -> int:
    """Number of GROUNDED evidence anchors in a research finding.

    A grounded anchor is an ``evidence`` entry that cites a concrete location —
    a non-empty ``path``. Unanchored prose contributes nothing (cite-or-discount):
    a side that asserts a conclusion without a single file:line anchor is treated
    as having no evidence, regardless of how confident its ``summary`` reads.
    """
    if not isinstance(finding, dict):
        return 0
    evidence = finding.get("evidence")
    if not isinstance(evidence, list):
        return 0
    return sum(
        1 for e in evidence
        if isinstance(e, dict) and str(e.get("path") or "").strip()
    )


def arbitrate(
    for_finding: dict[str, Any] | None,
    against_finding: dict[str, Any] | None,
) -> tuple[str, str, str]:
    """Deterministic three-state verdict from grounded-anchor counts.

    Returns ``(verdict, confidence, cell)`` where ``verdict`` is
    ``leans-for`` | ``leans-against`` | ``uncertain``; ``cell`` is the oracle-cell
    self-classification (``resolved`` | ``contested`` | ``insufficient_evidence``).

    The weighting is by GROUNDED evidence anchors only (cite-or-discount), never
    rhetoric. An unanchored side is discounted to zero, so a substantiated side
    wins over a merely fluent one. When neither side is substantiated, or both are
    substantiated equally, the honest answer is ``uncertain`` — a deliberation
    that always emits a confident verdict is dangerous for decisions.
    """
    f = count_grounded_anchors(for_finding)
    a = count_grounded_anchors(against_finding)

    if f == 0 and a == 0:
        return "uncertain", "low", "insufficient_evidence"
    if f > 0 and a == 0:
        conf = "high" if f >= 3 else ("medium" if f >= 2 else "low")
        return "leans-for", conf, "resolved"
    if a > 0 and f == 0:
        conf = "high" if a >= 3 else ("medium" if a >= 2 else "low")
        return "leans-against", conf, "resolved"

    # Both sides substantiated.
    margin = abs(f - a)
    if margin == 0:
        return "uncertain", "low", "contested"
    verdict = "leans-for" if f > a else "leans-against"
    conf = "medium" if margin >= 2 else "low"
    return verdict, conf, "resolved"


def aggregate(subclaims: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    """Roll sub-claim verdicts into ``(overall_verdict, confidence, uncertainties)``.

    ``overall_verdict`` ∈ ``leans-for`` | ``leans-against`` | ``mixed`` |
    ``uncertain``. ``uncertainties`` lists the claims that came back
    ``uncertain`` (the open questions a human must resolve).
    """
    verdicts = [str(s.get("verdict") or "uncertain") for s in subclaims]
    fors = verdicts.count("leans-for")
    againsts = verdicts.count("leans-against")

    if not verdicts:
        overall = "uncertain"
    elif fors and not againsts:
        overall = "leans-for"
    elif againsts and not fors:
        overall = "leans-against"
    elif fors and againsts:
        overall = "mixed"
    else:
        overall = "uncertain"

    resolved = fors + againsts
    if not verdicts:
        confidence = "low"
    else:
        frac = resolved / len(verdicts)
        confidence = "high" if frac >= 0.8 else ("medium" if frac >= 0.5 else "low")

    uncertainties = [
        str(s.get("claim") or "")
        for s in subclaims
        if str(s.get("verdict") or "") == "uncertain"
    ]
    return overall, confidence, uncertainties


def _empty_finding() -> dict[str, Any]:
    return {"summary": "", "evidence": [], "confidence": "low"}


_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _confidence_rank(confidence: Any) -> int:
    return _CONFIDENCE_RANK.get(str(confidence or "").lower(), 0)


def _merge_findings(acc: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Union two stance findings across adaptive-depth rounds.

    Evidence entries are de-duplicated by ``(path, lines, excerpt-prefix)`` so a
    later round repeating an earlier citation does NOT inflate the grounded-anchor
    count (that would let re-research fake progress). Summary prefers the latest
    non-empty; confidence takes the higher of the two.
    """
    seen: set[tuple[str, str, str]] = set()
    merged: list[dict[str, Any]] = []
    for entry in [*(acc.get("evidence") or []), *(new.get("evidence") or [])]:
        if not isinstance(entry, dict):
            continue
        key = (
            str(entry.get("path") or ""),
            str(entry.get("lines") or ""),
            str(entry.get("excerpt") or "")[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(entry)
    summary = str(new.get("summary") or "").strip() or str(acc.get("summary") or "")
    confidence = max(
        acc.get("confidence", "low"), new.get("confidence", "low"),
        key=_confidence_rank,
    )
    return {"summary": summary, "evidence": merged, "confidence": confidence}


def _deepen_hint(prior: dict[str, Any] | None) -> str:
    """Follow-up-round instruction: avoid repeating known anchors, dig for more."""
    anchors: list[str] = []
    for entry in (prior or {}).get("evidence") or []:
        if isinstance(entry, dict) and str(entry.get("path") or "").strip():
            anchors.append(f"{entry.get('path')}:{entry.get('lines')}")
    known = "; ".join(anchors) if anchors else "(none found yet)"
    return (
        "This is a FOLLOW-UP round — the first pass was inconclusive. Previously "
        f"found anchors: {known}. Find ADDITIONAL, DISTINCT grounded evidence not "
        "already listed (look in different files / from a different angle), or "
        "confirm none exists. Do not repeat the same citations."
    )


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class DeliberateRunner:
    """Run a single-round adversarial deliberation over a proposal.

    LLM surfaces: the decompose turn (``self.llm`` via ``_call_llm_bounded``) and
    the stance research workers (``ResearchRunner`` built on
    ``self.llm_provider``). Arbitration + aggregation are pure Python.

    Test seams: monkeypatch :meth:`_decompose` and :meth:`_research` to drive the
    pipeline with canned data and no LLM/engine.
    """

    cc_config: CCConfig
    llm_provider: Any  # LLMClientProvider — for ResearchRunner workers
    llm: LLMCallable  # for the decompose turn
    cwd: str
    language: str = "en"
    max_subclaims: int = 5
    max_rounds: int = 2
    """Adaptive-depth clamp (Phase 1). A sub-claim earns another adversarial
    round ONLY while it is thin/contested (verdict ``uncertain``); a resolved
    claim stops immediately. ``max_rounds=1`` ⇒ single-round (Phase-0 behaviour)."""
    llm_timeout_s: float = 600.0
    research_timeout_s: float = 600.0
    scope: str = ""
    max_tool_rounds: int | None = None

    def deliberate(self, proposal: str) -> dict[str, Any]:
        """Run the full pipeline (sync entry). Returns the INFORM-only result dict.

        Adaptive depth (Phase 1): each sub-claim runs round 1; if its verdict is
        ``uncertain`` (thin / contested) AND budget remains, it runs another round
        whose evidence accumulates into the same finding, then re-arbitrates. A
        ``resolved`` claim stops immediately. Hard-clamped by ``max_rounds``.
        """
        claims = self._decompose(proposal)
        subclaims: list[dict[str, Any]] = []
        max_rounds_used = 0
        budget = max(1, int(self.max_rounds))
        for claim in claims:
            for_acc = _empty_finding()
            against_acc = _empty_finding()
            verdict, confidence, cell = "uncertain", "low", "insufficient_evidence"
            rounds_used = 0
            for round_idx in range(budget):
                rounds_used = round_idx + 1
                for_new, against_new = self._research_pair(
                    claim, round_idx=round_idx,
                    prior_for=for_acc, prior_against=against_acc,
                )
                for_acc = _merge_findings(for_acc, for_new)
                against_acc = _merge_findings(against_acc, against_new)
                verdict, confidence, cell = arbitrate(for_acc, against_acc)
                # Only a thin/contested (uncertain) claim earns another round.
                if verdict != "uncertain":
                    break
            max_rounds_used = max(max_rounds_used, rounds_used)
            f_anchors = count_grounded_anchors(for_acc)
            a_anchors = count_grounded_anchors(against_acc)
            subclaims.append({
                "claim": claim,
                "for": {**for_acc, "anchors": f_anchors},
                "against": {**against_acc, "anchors": a_anchors},
                "verdict": verdict,
                "confidence": confidence,
                "cell": cell,
                "rounds": rounds_used,
                "rationale": (
                    f"support cited {f_anchors} grounded anchor(s); "
                    f"refute cited {a_anchors}; verdict={verdict} ({cell}); "
                    f"rounds={rounds_used}"
                ),
            })
        overall, overall_conf, uncertainties = aggregate(subclaims)
        return {
            "proposal": (proposal or "").strip(),
            "subclaims": subclaims,
            "overall_verdict": overall,
            "overall_confidence": overall_conf,
            "open_uncertainties": uncertainties,
            "rounds": max_rounds_used,
        }

    # -- decompose ----------------------------------------------------------- #

    def _decompose(self, proposal: str) -> list[str]:
        """Bounded LLM turn → sub-claims. Falls back to [proposal] on failure."""
        system = _DECOMPOSE_SYSTEM_PROMPT
        user = _build_decompose_user_prompt(proposal, self.max_subclaims)
        response = _bounded_llm_text(
            self.llm, system=system, user=user,
            purpose="deliberate.decompose", timeout_s=self.llm_timeout_s,
        )
        data = parse_llm_json(
            response, schema_name="deliberate_decompose",
            fallback_factory=lambda _raw: {}, expected_type=dict,
        )
        raw = data.get("subclaims")
        claims: list[str] = []
        if isinstance(raw, list):
            for c in raw:
                text = str(c).strip()
                if text:
                    claims.append(text)
        claims = claims[: self.max_subclaims]
        if not claims:
            fallback = (proposal or "").strip()
            return [fallback] if fallback else []
        return claims

    # -- adversarial research ------------------------------------------------ #

    def _research_pair(
        self, claim: str, *, round_idx: int = 0,
        prior_for: dict[str, Any] | None = None,
        prior_against: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run support + refute workers in parallel (separate contexts).

        On a follow-up round (``round_idx > 0``) each worker is told what its
        stance already found so it digs for distinct evidence (adaptive depth).
        """
        results: dict[str, dict[str, Any]] = {}
        priors = {"support": prior_for, "refute": prior_against}

        def _worker(stance: str) -> None:
            try:
                results[stance] = self._research(
                    claim, stance, round_idx=round_idx,
                    prior_evidence=priors.get(stance),
                )
            except Exception:  # noqa: BLE001 — a worker crash must not kill the run
                logger.warning(
                    "ccx deliberate: %s worker raised for claim %r",
                    stance, claim, exc_info=True,
                )
                results[stance] = _empty_finding()

        threads = [
            threading.Thread(
                target=_worker, args=(stance,),
                name=f"ccx-deliberate-{stance}", daemon=True,
            )
            for stance in ("support", "refute")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(self.research_timeout_s)
        for stance in ("support", "refute"):
            if stance not in results:
                logger.warning(
                    "ccx deliberate: %s worker did not return within %.0fs; "
                    "treating as no-evidence", stance, self.research_timeout_s,
                )
                results[stance] = _empty_finding()
        return results["support"], results["refute"]

    def _research(
        self, claim: str, stance: str, *, round_idx: int = 0,
        prior_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run ONE stance-committed read-only research worker. Test seam.

        Returns ``{summary, evidence:[{path,lines,excerpt}], confidence}`` — the
        ResearchRunner output, normalized. On a follow-up round
        (``round_idx > 0``) the goal carries a deepen-hint so the worker avoids
        repeating known anchors. Monkeypatch this in tests to inject canned
        findings without an engine/LLM.
        """
        goal = _RESEARCH_QUESTION.format(claim=claim)
        if round_idx > 0:
            goal = f"{goal}\n\n{_deepen_hint(prior_evidence)}"
        invocation = SubagentInvocation(
            goal=goal,
            mode="research",
            metadata={"scope": self.scope} if self.scope else {},
        )
        runner = ResearchRunner(
            cc_config=self.cc_config,
            llm_provider=self.llm_provider,
            cwd=self.cwd,
            max_tool_rounds=self.max_tool_rounds,
            stance=stance,
        )
        result = runner.run(invocation)
        extras = result.extras or {}
        evidence = extras.get("evidence")
        return {
            "summary": str(result.final_text or ""),
            "evidence": list(evidence) if isinstance(evidence, list) else [],
            "confidence": str(extras.get("confidence") or "low"),
        }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render_deliberation_report(deliberation: dict[str, Any]) -> str:
    """Render the deliberation dict as a human-readable markdown report."""
    d = deliberation or {}
    lines: list[str] = []
    lines.append("# Deliberation")
    lines.append("")
    lines.append(f"**Proposal:** {d.get('proposal') or '(empty)'}")
    lines.append("")
    lines.append(
        f"**Overall:** {d.get('overall_verdict', 'uncertain')} "
        f"(confidence: {d.get('overall_confidence', 'low')})"
    )
    lines.append("")
    for i, s in enumerate(d.get("subclaims") or [], 1):
        lines.append(f"## {i}. {s.get('claim', '')}")
        lines.append(
            f"- verdict: **{s.get('verdict', 'uncertain')}** "
            f"({s.get('confidence', 'low')}, {s.get('cell', '')})"
        )
        for side in ("for", "against"):
            f = s.get(side) or {}
            label = "FOR" if side == "for" else "AGAINST"
            lines.append(
                f"- {label} ({f.get('anchors', 0)} grounded anchor(s)): "
                f"{f.get('summary') or '(no evidence)'}"
            )
        lines.append("")
    uncertainties = d.get("open_uncertainties") or []
    if uncertainties:
        lines.append("## Open uncertainties (need a human / new evidence)")
        for u in uncertainties:
            lines.append(f"- {u}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
