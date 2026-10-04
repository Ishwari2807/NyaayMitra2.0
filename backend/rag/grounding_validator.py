from __future__ import annotations

from typing import Any, Dict, List

from backend.rag.claim_support import ClaimSupportChecker


class GroundingValidator:
    """
    Claim-level post-generation grounding validator.

    Important:
    - Sources remain deterministic retrieval metadata.
    - Generated structured sections are preserved when they are
      generated from approved evidence.
    - Unsupported individual claims are not allowed to replace an
      entire useful section with a single fallback sentence.
    """

    TEXT_FIELDS = [
        "issue_detected",
        "situation",
        "what_may_be_relevant",
        "what_rights_may_apply",
        "recommended_next_steps",
        "evidence_checklist",
        "legal_area",
        "responsibilities",
        "evidence",
        "next_steps",
        "authority",
    ]

    # These are presentation/interpretation fields. We do not want
    # aggressive lexical filtering to collapse them into one line.
    STRUCTURED_FIELDS = {
        "what_may_be_relevant",
        "what_rights_may_apply",
        "recommended_next_steps",
        "evidence_checklist",
        "responsibilities",
        "authority",
        "next_steps",
        "evidence",
    }

    def __init__(
        self,
        support_checker: ClaimSupportChecker | None = None,
    ):
        self.support_checker = (
            support_checker
            or ClaimSupportChecker()
        )

    def validate(
        self,
        response: Dict[str, Any],
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not response:
            return self._reject(
                response,
                "The generated response was empty.",
            )

        if not evidence:
            return self._reject(
                response,
                "No verified evidence was available for grounding.",
            )

        result = dict(response)

        warnings: List[Dict[str, Any]] = []

        substantive_checked = 0
        unsupported_count = 0

        for field in self.TEXT_FIELDS:

            value = result.get(field)

            cleaned, field_warnings, checked, unsupported = (
                self._validate_value(
                    field,
                    value,
                    evidence,
                )
            )

            result[field] = cleaned

            warnings.extend(
                field_warnings
            )

            substantive_checked += checked
            unsupported_count += unsupported

        # ---------------------------------------------------------
        # Status handling
        # ---------------------------------------------------------

        if (
            unsupported_count
            and substantive_checked > 0
            and unsupported_count >= substantive_checked
        ):
            result["status"] = (
                "INSUFFICIENT_INFORMATION"
            )

        elif unsupported_count:
            result["status"] = (
                "PARTIALLY_SUPPORTED"
            )

        # ---------------------------------------------------------
        # Warnings
        # ---------------------------------------------------------

        if warnings:

            result["grounding_warnings"] = warnings

            limitations = (
                result.get("limitations")
                if isinstance(
                    result.get("limitations"),
                    list,
                )
                else []
            )

            msg = (
                "Some generated claims could not be strongly "
                "matched to the verified evidence."
            )

            if msg not in limitations:
                limitations.append(msg)

            result["limitations"] = limitations

        return result

    def _validate_value(
        self,
        field: str,
        value: Any,
        evidence: List[Dict[str, Any]],
    ):

        warnings: List[Dict[str, Any]] = []

        if value is None:
            return (
                value,
                warnings,
                0,
                0,
            )

        # ---------------------------------------------------------
        # Strings
        # ---------------------------------------------------------

        if isinstance(value, str):

            if not value.strip():
                return (
                    value,
                    warnings,
                    0,
                    0,
                )

            score = self._support_score(
                value,
                evidence,
            )

            # Keep the main situation/issue unless it is completely
            # unrelated. We do not delete it merely because the
            # wording differs from the source text.
            if score <= 0.05:

                warnings.append(
                    {
                        "field": field,
                        "claim": value,
                        "score": score,
                    }
                )

                # Do not destroy the entire field.
                return (
                    value,
                    warnings,
                    1,
                    0,
                )

            return (
                value,
                warnings,
                1,
                0,
            )

        # ---------------------------------------------------------
        # Lists
        # ---------------------------------------------------------

        if isinstance(value, list):

            if not value:
                return (
                    value,
                    warnings,
                    0,
                    0,
                )

            cleaned_items = []

            for item in value:

                if isinstance(item, dict):

                    # Applicable-law objects and other structured
                    # objects are retained. Their contents have already
                    # been generated from approved evidence.
                    cleaned_items.append(item)

                    continue

                if not isinstance(item, str):
                    continue

                item = item.strip()

                if not item:
                    continue

                score = self._support_score(
                    item,
                    evidence,
                )

                # -------------------------------------------------
                # IMPORTANT CHANGE:
                #
                # Do NOT remove structured generated items merely
                # because lexical overlap is low.
                #
                # The LLM may legitimately paraphrase legal text.
                # The V2 design explicitly allows grounded explanation
                # rather than requiring exact wording.
                # -------------------------------------------------

                if field in self.STRUCTURED_FIELDS:

                    cleaned_items.append(item)

                    if score <= 0.05:

                        warnings.append(
                            {
                                "field": field,
                                "claim": item,
                                "score": score,
                            }
                        )

                    continue

                # For non-structured fields, retain supported claims.
                if score > 0.05:

                    cleaned_items.append(item)

                else:

                    warnings.append(
                        {
                            "field": field,
                            "claim": item,
                            "score": score,
                        }
                    )

            return (
                cleaned_items,
                warnings,
                len(cleaned_items),
                0,
            )

        # ---------------------------------------------------------
        # Dictionaries / structured objects
        # ---------------------------------------------------------

        if isinstance(value, dict):

            return (
                value,
                warnings,
                1,
                0,
            )

        return (
            value,
            warnings,
            0,
            0,
        )

    def _support_score(
        self,
        claim: str,
        evidence: List[Dict[str, Any]],
    ) -> float:

        best_score = 0.0

        for item in evidence:

            try:

                result = self.support_checker.check(
                    claim,
                    item,
                )

                if isinstance(
                    result,
                    dict,
                ):

                    score = float(
                        result.get(
                            "score",
                            0.0,
                        )
                    )

                else:

                    score = float(
                        result
                    )

                best_score = max(
                    best_score,
                    score,
                )

            except Exception:

                # Validator must never break the pipeline.
                continue

        return best_score

    def _reject(
        self,
        response: Dict[str, Any] | None,
        message: str,
    ) -> Dict[str, Any]:

        result = dict(
            response or {}
        )

        result["status"] = (
            "INSUFFICIENT_INFORMATION"
        )

        limitations = (
            result.get("limitations")
            if isinstance(
                result.get("limitations"),
                list,
            )
            else []
        )

        limitations.append(
            message
        )

        result["limitations"] = limitations

        return result