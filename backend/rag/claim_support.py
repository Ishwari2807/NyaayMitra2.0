from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List


class ClaimSupportChecker:
    """Lightweight claim-level verifier for generated structured answers.

    It checks each substantive string/list item independently against verified
    evidence. This is deliberately conservative and provider-independent; a
    future semantic claim checker can replace this class without changing the
    orchestration contract.
    """

    STOPWORDS = {"the", "and", "for", "that", "with", "from", "this", "may", "can", "are", "is", "to", "of", "in", "on", "a", "an", "be", "by"}

    @classmethod
    def tokens(cls, text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2 and t not in cls.STOPWORDS}

    @classmethod
    def score(cls, claim: str, evidence: str) -> float:
        ct = cls.tokens(claim)
        et = cls.tokens(evidence)
        if not ct:
            return 1.0
        overlap = len(ct & et) / len(ct)
        sequence = SequenceMatcher(None, claim.lower(), evidence.lower()).ratio()
        refs = re.findall(r"\b(?:section|article|rule|regulation)\s+\d+[a-zA-Z0-9()\-]*", claim.lower())
        ref_support = any(ref in evidence.lower() for ref in refs) if refs else False
        return max(overlap, sequence, 0.90 if ref_support else 0.0)

    def check(self, claim: str, evidence_items: Iterable[Dict[str, Any]], threshold: float = 0.34) -> Dict[str, Any]:
        evidence = [item.get("text", "") for item in evidence_items if item.get("text")]
        if not evidence:
            return {"supported": False, "score": 0.0, "evidence_indexes": []}
        scores = [self.score(claim, text) for text in evidence]
        best = max(scores)
        indexes = [i for i, value in enumerate(scores) if value == best and value >= threshold]
        return {"supported": bool(indexes), "score": float(best), "evidence_indexes": indexes}
