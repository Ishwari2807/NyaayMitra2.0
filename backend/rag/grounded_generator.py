from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List

from backend.rag.grounding_validator import GroundingValidator
from backend.rag.grounded_prompt import (
    SYSTEM_PROMPT,
    build_grounded_prompt,
)


ALLOWED_STATUSES = {
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "INSUFFICIENT_INFORMATION",
    "NO_RELEVANT_SOURCE",
}


def _extract_json_object(raw_response: str) -> str:

    if not raw_response:
        return ""

    raw_response = raw_response.strip()

    raw_response = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        raw_response,
        flags=re.IGNORECASE,
    )

    start = raw_response.find("{")
    end = raw_response.rfind("}")

    if start == -1 or end == -1 or end < start:
        return raw_response

    return raw_response[start:end + 1]


def _build_sources(
    evidence: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    sources = []

    for item in evidence:

        metadata = item.get(
            "metadata",
            {},
        ) or {}

        source = {
            "chunk_id": item.get(
                "chunk_id",
                "",
            ),
            "source_file": metadata.get(
                "source_file",
                "",
            ),
            "section_numbers": metadata.get(
                "section_numbers",
                [],
            ),
        }

        if metadata.get("page"):
            source["page"] = metadata.get("page")

        sources.append(source)

    return sources


def _is_empty(value: Any) -> bool:

    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    if isinstance(value, list):
        return len(value) == 0

    if isinstance(value, dict):
        return len(value) == 0

    return False


class GroundedAnswerGenerator:

    def __init__(
        self,
        llm_generate: Callable[[str, str], str],
    ):

        self.llm_generate = llm_generate
        self.grounding_validator = GroundingValidator()

    def generate(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not query or not query.strip():

            return {
                "status": "INSUFFICIENT_INFORMATION",
                "error": "Empty query.",
            }

        if not evidence:

            return {
                "status": "INSUFFICIENT_INFORMATION",
                "error": "No verified evidence was supplied.",
            }

        prompt = build_grounded_prompt(
            query=query,
            evidence=evidence,
        )

        raw_response = self.llm_generate(
            SYSTEM_PROMPT,
            prompt,
        )

        cleaned_response = _extract_json_object(
            raw_response
        )

        try:

            response = json.loads(
                cleaned_response
            )

        except json.JSONDecodeError:

            return {
                "status": "GENERATION_ERROR",
                "error": "The language model did not return valid JSON.",
                "raw_response": raw_response,
            }

        if not isinstance(response, dict):

            return {
                "status": "GENERATION_ERROR",
                "error": "The language model response was not a JSON object.",
                "raw_response": raw_response,
            }

        return self._validate_and_finalize(
            response=response,
            evidence=evidence,
            raw_response=raw_response,
        )

    def _validate_and_finalize(
        self,
        response: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        raw_response: str,
    ) -> Dict[str, Any]:

        result: Dict[str, Any] = {}

        result["status"] = str(
            response.get(
                "status",
                "INSUFFICIENT_INFORMATION",
            )
        ).strip().upper()

        if result["status"] not in ALLOWED_STATUSES:
            result["status"] = "INSUFFICIENT_INFORMATION"

        result["situation"] = response.get(
            "situation",
            "",
        )

        result["legal_area"] = response.get(
            "legal_area",
            [],
        )

        result["applicable_law"] = response.get(
            "applicable_law",
            [],
        )

        result["rights"] = response.get(
            "rights",
            [],
        )

        result["responsibilities"] = response.get(
            "responsibilities",
            [],
        )

        result["evidence_checklist"] = response.get(
            "evidence_checklist",
            [],
        )

        result["recommended_next_steps"] = response.get(
            "recommended_next_steps",
            [],
        )

        result["authority"] = response.get(
            "authority",
            [],
        )

        result["important_note"] = response.get(
            "important_note",
            "",
        )

        result["sources"] = _build_sources(
            evidence
        )

        # ---------------------------------------------------------
        # Normalize list fields
        # ---------------------------------------------------------

        list_fields = [
            "legal_area",
            "applicable_law",
            "rights",
            "responsibilities",
            "evidence_checklist",
            "recommended_next_steps",
            "authority",
        ]

        for field in list_fields:

            if not isinstance(
                result.get(field),
                list,
            ):
                result[field] = []

        # ---------------------------------------------------------
        # Normalize applicable law
        # ---------------------------------------------------------

        normalized_law = []

        for item in result["applicable_law"]:

            if isinstance(item, dict):

                normalized_law.append(
                    {
                        "law_name": str(
                            item.get(
                                "law_name",
                                "",
                            )
                        ).strip(),

                        "provision": str(
                            item.get(
                                "provision",
                                "Not specified in retrieved evidence",
                            )
                        ).strip(),

                        "explanation": str(
                            item.get(
                                "explanation",
                                "",
                            )
                        ).strip(),

                        "relevance": str(
                            item.get(
                                "relevance",
                                "",
                            )
                        ).strip(),
                    }
                )

            elif isinstance(item, str):

                normalized_law.append(
                    {
                        "law_name": item.strip(),
                        "provision": "Not specified in retrieved evidence",
                        "explanation": "",
                        "relevance": "",
                    }
                )

        result["applicable_law"] = normalized_law

        # ---------------------------------------------------------
        # Backward-compatible aliases
        # ---------------------------------------------------------

        result["evidence"] = list(
            result.get(
                "evidence_checklist",
                [],
            )
        )

        result["next_steps"] = list(
            result.get(
                "recommended_next_steps",
                [],
            )
        )

        # ---------------------------------------------------------
        # Grounding validation
        # ---------------------------------------------------------

        validated = self.grounding_validator.validate(
            response=result,
            evidence=evidence,
        )

        # ---------------------------------------------------------
        # IMPORTANT:
        # Keep structured legal fields even when lexical grounding
        # score is low. They have already been generated from the
        # approved evidence.
        # ---------------------------------------------------------

        if not isinstance(
            validated.get("applicable_law"),
            list,
        ):
            validated["applicable_law"] = result["applicable_law"]

        if not isinstance(
            validated.get("rights"),
            list,
        ):
            validated["rights"] = result["rights"]

        if not isinstance(
            validated.get("responsibilities"),
            list,
        ):
            validated["responsibilities"] = result["responsibilities"]

        if not isinstance(
            validated.get("authority"),
            list,
        ):
            validated["authority"] = result["authority"]

        if not isinstance(
            validated.get("evidence_checklist"),
            list,
        ):
            validated["evidence_checklist"] = result[
                "evidence_checklist"
            ]

        if not isinstance(
            validated.get("recommended_next_steps"),
            list,
        ):
            validated["recommended_next_steps"] = result[
                "recommended_next_steps"
            ]

        if not isinstance(
            validated.get("legal_area"),
            list,
        ):
            validated["legal_area"] = result["legal_area"]

        validated["evidence"] = validated.get(
            "evidence_checklist",
            result["evidence_checklist"],
        )

        validated["next_steps"] = validated.get(
            "recommended_next_steps",
            result["recommended_next_steps"],
        )

        validated["sources"] = _build_sources(
            evidence
        )

        return validated