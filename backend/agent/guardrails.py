"""Deterministic safety and grounding checks for retrieved evidence."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

from backend.models import Citation, SearchResult


_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore_instructions", re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I)),
    ("reveal_prompt", re.compile(r"(reveal|print|show).{0,24}(system|developer)\s+prompt", re.I)),
    ("override_role", re.compile(r"you\s+are\s+now\b|act\s+as\s+(an?|the)\b", re.I)),
    ("ignore_instructions_zh", re.compile(r"忽略.{0,12}(之前|以上|前面).{0,8}(指令|要求|提示词)")),
    ("reveal_prompt_zh", re.compile(r"(输出|泄露|显示).{0,12}(系统提示词|开发者指令|隐藏提示词)")),
)
_CITATION_RE = re.compile(r"\[(\d+)]")


@dataclass(slots=True)
class CitationAudit:
    """Machine-readable answer grounding report used by the critic and evals."""

    valid_indices: list[int] = field(default_factory=list)
    invalid_indices: list[int] = field(default_factory=list)
    claim_count: int = 0
    cited_claim_count: int = 0
    citation_coverage: float = 0.0
    lexical_grounding: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.invalid_indices and (self.claim_count == 0 or self.citation_coverage >= 0.5)


def sanitize_evidence(text: str) -> tuple[str, list[str]]:
    """Remove instruction-like lines from untrusted documents before prompting.

    Knowledge-base documents are data, not instructions.  We keep ordinary
    content intact and replace only lines matching explicit prompt-injection
    signatures.  The flags are persisted in the run trace and force review.
    """

    flags: list[str] = []
    safe_lines: list[str] = []
    normalized = text.replace("\u200b", "").replace("\ufeff", "")
    for line in normalized.splitlines() or [normalized]:
        matched = [name for name, pattern in _INJECTION_PATTERNS if pattern.search(line)]
        if matched:
            flags.extend(matched)
            safe_lines.append("[已移除疑似提示注入指令]")
        else:
            safe_lines.append(line)
    return "\n".join(safe_lines), list(dict.fromkeys(flags))


def retrieval_confidence(results: list[SearchResult]) -> float:
    """Estimate confidence from absolute retrieval signals, not rank-normalized scores.

    The old implementation normalized the best result to one, making even one
    arbitrary hit look perfectly confident.  This heuristic combines cosine
    similarity, saturating BM25 evidence, the top-result margin and evidence
    agreement.  It is intentionally transparent and should later be calibrated
    against the repository's eval set.
    """

    if not results:
        return 0.0

    signals: list[float] = []
    for result in results[:3]:
        semantic = max(0.0, min(1.0, float(result.vector_score)))
        lexical = 1.0 - math.exp(-max(0.0, float(result.bm25_score)))
        signals.append(0.6 * semantic + 0.4 * lexical)

    top = signals[0]
    runner_up = signals[1] if len(signals) > 1 else 0.0
    margin = max(0.0, top - runner_up)
    agreement = sum(signals) / len(signals)
    confidence = 0.65 * top + 0.25 * agreement + 0.10 * margin
    return round(max(0.0, min(1.0, confidence)), 6)


def audit_citations(answer: str, sources: list[Citation], results: list[SearchResult]) -> CitationAudit:
    """Validate citation indices and estimate claim-level evidence coverage."""

    indices = [int(value) for value in _CITATION_RE.findall(answer)]
    valid = sorted({value for value in indices if 1 <= value <= len(sources)})
    invalid = sorted({value for value in indices if value < 1 or value > len(sources)})

    claims = [
        part.strip()
        for part in re.split(r"(?<=[。！？!?；;])|\n+", answer)
        if len(_CITATION_RE.sub("", part).strip()) >= 8
    ]
    cited_claims = [claim for claim in claims if _CITATION_RE.search(claim)]
    coverage = len(cited_claims) / len(claims) if claims else (1.0 if not sources else 0.0)

    evidence_by_source: dict[int, str] = {}
    for index, source in enumerate(sources, 1):
        texts = [
            item.chunk.text
            for item in results
            if item.chunk.source_path == source.path and item.chunk.page == source.page
        ]
        evidence_by_source[index] = " ".join(texts) or source.snippet

    support_scores: list[float] = []
    for claim in cited_claims:
        cited = [int(value) for value in _CITATION_RE.findall(claim) if 1 <= int(value) <= len(sources)]
        claim_tokens = _tokens(_CITATION_RE.sub("", claim))
        evidence_tokens = set()
        for index in cited:
            evidence_tokens.update(_tokens(evidence_by_source.get(index, "")))
        if claim_tokens:
            support_scores.append(len(claim_tokens & evidence_tokens) / len(claim_tokens))
    grounding = sum(support_scores) / len(support_scores) if support_scores else 0.0

    return CitationAudit(
        valid_indices=valid,
        invalid_indices=invalid,
        claim_count=len(claims),
        cited_claim_count=len(cited_claims),
        citation_coverage=round(coverage, 6),
        lexical_grounding=round(grounding, 6),
    )


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    english = set(re.findall(r"[a-z0-9_\-]{2,}", lowered))
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    chinese = {
        run[index:index + 2]
        for run in chinese_runs
        for index in range(max(1, len(run) - 1))
        if run[index:index + 2]
    }
    return english | chinese
