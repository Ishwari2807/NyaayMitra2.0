from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


class EvidenceGate:
    """
    Decide whether verified evidence is sufficient for grounded generation.

    The gate checks:
        1. Retrieved evidence exists.
        2. Evidence has valid provenance.
        3. Evidence is relevant to the user's information needs.
        4. Identified issues have reasonable evidence coverage.

    It does not decide the legal outcome.
    """

    STATUSES = {
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "INSUFFICIENT_INFORMATION",
        "NO_RELEVANT_SOURCE",
    }

    DEFINITION_TERMS = {
        "definition",
        "define",
        "defined",
        "meaning",
        "what is",
        "what are",
    }

    def __init__(
        self,
        min_candidates: int = 1,
        max_evidence_gap: float = 2.0,
        min_issue_coverage: float = 0.30,
        min_relevance_score: float = 0.20,
    ):
        self.min_candidates = min_candidates
        self.max_evidence_gap = max_evidence_gap
        self.min_issue_coverage = min_issue_coverage
        self.min_relevance_score = min_relevance_score

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(
                r"[a-z0-9]+",
                (text or "").lower(),
            )
            if len(token) > 2
        }

    # ------------------------------------------------------------------
    # Lexical coverage
    # ------------------------------------------------------------------

    @classmethod
    def _coverage(
        cls,
        need: str,
        evidence_text: str,
    ) -> float:

        need_tokens = cls._tokens(need)

        if not need_tokens:
            return 1.0

        evidence_tokens = cls._tokens(
            evidence_text
        )

        return (
            len(
                need_tokens & evidence_tokens
            )
            / len(need_tokens)
        )

    # ------------------------------------------------------------------
    # Source validation
    # ------------------------------------------------------------------

    @staticmethod
    def _source_valid(
        result: Dict[str, Any],
    ) -> bool:

        metadata = (
            result.get("metadata")
            or {}
        )

        return bool(
            result.get("chunk_id")
            and result.get("text")
            and (
                metadata.get("source_file")
                or metadata.get("document_id")
            )
        )

    # ------------------------------------------------------------------
    # Definition query detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_definition_query(
        query: str,
    ) -> bool:

        q = (
            query or ""
        ).strip().lower()

        return any(
            term in q
            for term in EvidenceGate.DEFINITION_TERMS
        )

    # ------------------------------------------------------------------
    # Reranker relevance
    # ------------------------------------------------------------------

    @staticmethod
    def _has_relevant_reranked_evidence(
        results: List[Dict[str, Any]],
        minimum_score: float,
    ) -> bool:

        scores = [
            float(result["reranker_score"])
            for result in results
            if result.get("reranker_score") is not None
        ]

        if not scores:
            return False

        return max(scores) >= minimum_score

    # ------------------------------------------------------------------
    # Select evidence
    # ------------------------------------------------------------------

    def select_evidence(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        if not results:
            return []

        top_score = results[0].get(
            "reranker_score"
        )

        if top_score is None:
            return results

        selected = []

        for result in results:

            score = result.get(
                "reranker_score"
            )

            if (
                score is not None
                and float(top_score)
                - float(score)
                <= self.max_evidence_gap
            ):
                selected.append(result)

        return (
            selected
            or [results[0]]
        )

    # ------------------------------------------------------------------
    # Determine information needs
    # ------------------------------------------------------------------

    @staticmethod
    def _build_information_needs(
        query: str,
        situation: Dict[str, Any],
        information_needs: Optional[Iterable[str]],
    ) -> List[str]:

        explicit_needs = list(
            information_needs or []
        )

        if explicit_needs:
            return [
                need.strip()
                for need in explicit_needs
                if isinstance(need, str)
                and need.strip()
            ]

        situation_needs = list(
            situation.get(
                "retrieval_information_needs",
                [],
            )
            or []
        )

        if situation_needs:
            return [
                need.strip()
                for need in situation_needs
                if isinstance(need, str)
                and need.strip()
            ]

        issues = list(
            situation.get(
                "potential_issues",
                [],
            )
            or []
        )

        if issues:
            return [
                issue.strip()
                for issue in issues
                if isinstance(issue, str)
                and issue.strip()
            ]

        # Last-resort fallback.
        # Do NOT use the whole natural-language query as a strict
        # legal coverage requirement. Instead, extract meaningful
        # terms from it.
        query_tokens = sorted(
            EvidenceGate._tokens(query)
        )

        if query_tokens:
            return [
                " ".join(query_tokens)
            ]

        return []

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query: str,
        results: List[Dict[str, Any]],
        situation: Optional[Dict[str, Any]] = None,
        information_needs: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:

        if not results:

            return {
                "status": "NO_RELEVANT_SOURCE",
                "proceed": False,
                "reason": (
                    "No verified source was retrieved."
                ),
                "evidence": [],
                "coverage": {},
            }

        if len(results) < self.min_candidates:

            return {
                "status": "INSUFFICIENT_INFORMATION",
                "proceed": False,
                "reason": (
                    "Too little verified evidence "
                    "was retrieved."
                ),
                "evidence": results,
                "coverage": {},
            }

        valid_results = [
            result
            for result in results
            if self._source_valid(result)
        ]

        if not valid_results:

            return {
                "status": "NO_RELEVANT_SOURCE",
                "proceed": False,
                "reason": (
                    "Retrieved items do not have "
                    "valid provenance."
                ),
                "evidence": [],
                "coverage": {},
            }

        situation = (
            situation
            if isinstance(situation, dict)
            else {}
        )

        needs = self._build_information_needs(
            query=query,
            situation=situation,
            information_needs=information_needs,
        )

        evidence_text = "\n".join(
            result.get("text", "")
            for result in valid_results
        )

        # --------------------------------------------------------------
        # Information-need coverage
        # --------------------------------------------------------------

        coverage = {
            need: self._coverage(
                need,
                evidence_text,
            )
            for need in needs
        }

        covered = [
            need
            for need, score in coverage.items()
            if score >= self.min_issue_coverage
        ]

        coverage_ratio = (
            len(covered) / len(needs)
            if needs
            else 0.0
        )

        # --------------------------------------------------------------
        # Reranker relevance
        # --------------------------------------------------------------

        has_reranker_relevance = (
            self._has_relevant_reranked_evidence(
                valid_results,
                self.min_relevance_score,
            )
        )

        selected = self.select_evidence(
            valid_results
        )

        # --------------------------------------------------------------
        # Decision
        # --------------------------------------------------------------

        if coverage_ratio >= 1.0:

            status = "SUPPORTED"
            proceed = True

            reason = (
                "Verified evidence sufficiently "
                "covers the identified information needs."
            )

        elif coverage_ratio > 0:

            status = "PARTIALLY_SUPPORTED"
            proceed = True

            reason = (
                "Verified evidence covers some, "
                "but not all, identified information needs."
            )

        elif has_reranker_relevance:

            # The evidence is strongly relevant according to the
            # cross-encoder even though lexical overlap is limited.
            # Allow grounded generation, where the generator and
            # claim-level validator provide the next safety layer.

            status = "PARTIALLY_SUPPORTED"
            proceed = True

            reason = (
                "Retrieved evidence is strongly relevant "
                "but lexical coverage of the information needs "
                "is incomplete."
            )

        else:

            status = "INSUFFICIENT_INFORMATION"
            proceed = False

            reason = (
                "Verified evidence does not sufficiently "
                "cover the user's legal information needs."
            )

        # --------------------------------------------------------------
        # Definition queries remain conservative
        # --------------------------------------------------------------

        if (
            self._is_definition_query(query)
            and coverage_ratio == 0
            and not has_reranker_relevance
        ):

            status = "INSUFFICIENT_INFORMATION"
            proceed = False

            reason = (
                "The retrieved sources do not contain "
                "sufficient evidence for the requested "
                "legal concept."
            )

        return {
            "status": status,
            "proceed": proceed,
            "reason": reason,
            "evidence": selected,
            "coverage": coverage,
            "issues_covered": covered,
            "issues_uncovered": [
                need
                for need in needs
                if need not in covered
            ],
            "source_validity": all(
                self._source_valid(result)
                for result in selected
            ),
            "relevance_detected": has_reranker_relevance,
        }