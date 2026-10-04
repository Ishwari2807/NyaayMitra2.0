import json
from typing import Any, Callable, Dict


class SituationAnalyzer:
    """
    Generic situation-understanding component for NyaayMitra.

    Design principles:
    - The citizen's original input is the only factual source.
    - No legal/domain-specific rules are hardcoded.
    - The LLM is used for language understanding, not legal reasoning.
    - Unsupported information is rejected during validation.
    """

    EXTRACTION_SYSTEM_PROMPT = """
You are the factual extraction component of NyaayMitra.

Your task is to extract ONLY information explicitly stated in the
citizen's message.

The citizen's message is the ONLY source of facts.

IMPORTANT:
Do not use general knowledge.
Do not complete the story.
Do not guess what probably happened.
Do not infer facts merely because they are common or likely.

Do NOT:
- provide legal advice
- identify laws or Acts
- identify legal sections
- identify offences
- determine legal rights
- determine authorities
- determine remedies
- determine penalties
- determine jurisdiction
- invent events
- invent documents
- invent dates
- invent locations
- invent people
- invent conversations
- invent reasons
- invent actions taken by the citizen

If something is not explicitly stated, leave it out.

Preserve the meaning of the citizen's original words.

Extract the following:

{
  "facts": [],
  "people_or_entities_involved": [],
  "actions_or_events": [],
  "location": "",
  "time_information": "",
  "documents_or_evidence_mentioned": [],
  "important_terms": []
}

Rules:

- "facts": only facts explicitly stated.
- "people_or_entities_involved": only people, organizations,
  institutions, companies, services, or entities explicitly mentioned.
- "actions_or_events": only actions/events explicitly described.
- "location": only an explicitly stated location.
- "time_information": only explicitly stated time information.
- "documents_or_evidence_mentioned": only evidence/documents explicitly
  mentioned by the citizen.
- "important_terms": useful words or short phrases from the citizen's
  message. Prefer wording actually present in the message.

Return ONLY valid JSON.
"""


    INTERPRETATION_SYSTEM_PROMPT = """
You are the situation-structuring component of NyaayMitra.

You receive:

1. The ORIGINAL citizen message.
2. A FACTUAL EXTRACTION produced from that message.

Your task is to create a concise structured representation of the
citizen's situation.

The ORIGINAL citizen message remains the ultimate source of truth.

Do NOT add facts.

Do NOT use outside knowledge.

Do NOT determine:
- applicable laws
- Acts
- legal sections
- offences
- legal rights
- authorities
- remedies
- penalties
- legal conclusions

"problem_type" must be a neutral description of the situation,
not a legal classification.

"situation_summary" must summarize only what the citizen actually said.

"missing_information" should contain information that is genuinely
absent from the citizen's message and would materially help understand
the situation.

Do not list information as missing if it was already provided.

Do not invent answers to missing information.

Return ONLY valid JSON in exactly this structure:

{
  "situation_summary": "",
  "problem_type": "",
  "potential_issues": [],
  "retrieval_information_needs": [],
  "missing_information": []
}
"""


    REQUIRED_EXTRACTION_FIELDS = {
        "facts",
        "people_or_entities_involved",
        "actions_or_events",
        "location",
        "time_information",
        "documents_or_evidence_mentioned",
        "important_terms",
    }

    REQUIRED_INTERPRETATION_FIELDS = {
        "situation_summary",
        "problem_type",
        "potential_issues",
        "retrieval_information_needs",
        "missing_information",
    }

    LIST_EXTRACTION_FIELDS = {
        "facts",
        "people_or_entities_involved",
        "actions_or_events",
        "documents_or_evidence_mentioned",
        "important_terms",
    }


    def __init__(self, llm_generate: Callable[[str, str], str]):
        self.llm_generate = llm_generate


    def analyze(self, user_input: str) -> Dict[str, Any]:
        """
        Run the two-stage situation analysis.
        """

        if not user_input or not user_input.strip():
            return {
                "status": "INVALID_INPUT",
                "situation_summary": "",
                "problem_type": "",
                "potential_issues": [],
                "retrieval_information_needs": [],
                "facts": [],
                "people_or_entities_involved": [],
                "actions_or_events": [],
                "location": "",
                "time_information": "",
                "documents_or_evidence_mentioned": [],
                "important_terms": [],
                "missing_information": [],
            }

        original_input = user_input.strip()

        # ---------------------------------------------------------
        # Stage 1: Literal factual extraction
        # ---------------------------------------------------------

        try:
            extraction_response = self.llm_generate(
                self.EXTRACTION_SYSTEM_PROMPT,
                original_input,
            )
        except Exception as exc:
            return {
                "status": "GENERATION_ERROR",
                "stage": "EXTRACTION",
                "error": str(exc),
            }

        extraction_json = self._clean_json(extraction_response)

        try:
            extracted = json.loads(extraction_json)
        except (json.JSONDecodeError, TypeError):
            return {
                "status": "GENERATION_ERROR",
                "stage": "EXTRACTION",
                "error": "Factual extraction did not return valid JSON.",
                "raw_response": extraction_response,
            }

        extraction_error = self._validate_extraction(extracted)

        if extraction_error:
            return {
                "status": "INVALID_STRUCTURE",
                "stage": "EXTRACTION",
                "error": extraction_error,
                "raw_result": extracted,
                "raw_response": extraction_response,
            }

        extracted = self._normalize_extraction(extracted)

        # ---------------------------------------------------------
        # Stage 2: Situation interpretation
        # ---------------------------------------------------------

        interpretation_prompt = self._build_interpretation_prompt(
            original_input,
            extracted,
        )

        try:
            interpretation_response = self.llm_generate(
                self.INTERPRETATION_SYSTEM_PROMPT,
                interpretation_prompt,
            )
        except Exception as exc:
            return {
                "status": "GENERATION_ERROR",
                "stage": "INTERPRETATION",
                "error": str(exc),
            }

        interpretation_json = self._clean_json(
            interpretation_response
        )

        try:
            interpretation = json.loads(interpretation_json)
        except (json.JSONDecodeError, TypeError):
            return {
                "status": "GENERATION_ERROR",
                "stage": "INTERPRETATION",
                "error": (
                    "Situation interpretation did not return "
                    "valid JSON."
                ),
                "raw_response": interpretation_response,
            }

        interpretation_error = self._validate_interpretation(
            interpretation
        )

        if interpretation_error:
            return {
                "status": "INVALID_STRUCTURE",
                "stage": "INTERPRETATION",
                "error": interpretation_error,
                "raw_result": interpretation,
            }

        interpretation = self._normalize_interpretation(
            interpretation
        )

        # ---------------------------------------------------------
        # Generic grounding validation
        # ---------------------------------------------------------

        grounding_result = self._validate_grounding(
            original_input,
            extracted,
        )

        if not grounding_result["valid"]:
            return {
                "status": "GROUNDING_ERROR",
                "stage": "VALIDATION",
                "error": grounding_result["error"],
                "extracted_result": extracted,
            }

        # ---------------------------------------------------------
        # Final result
        # ---------------------------------------------------------

        return {
            "status": "OK",
            "situation_summary": interpretation[
                "situation_summary"
            ],
            "problem_type": interpretation[
                "problem_type"
            ],
            "potential_issues": interpretation["potential_issues"],
            "retrieval_information_needs": interpretation["retrieval_information_needs"],
            "facts": extracted["facts"],
            "people_or_entities_involved": extracted[
                "people_or_entities_involved"
            ],
            "actions_or_events": extracted[
                "actions_or_events"
            ],
            "location": extracted["location"],
            "time_information": extracted[
                "time_information"
            ],
            "documents_or_evidence_mentioned": extracted[
                "documents_or_evidence_mentioned"
            ],
            "important_terms": extracted[
                "important_terms"
            ],
            "missing_information": interpretation[
                "missing_information"
            ],
        }


    def _build_interpretation_prompt(
        self,
        original_input: str,
        extracted: Dict[str, Any],
    ) -> str:
        """
        Build the second-stage prompt.

        The original citizen message is always included so the
        interpretation stage cannot rely exclusively on the first
        model output.
        """

        payload = {
            "original_citizen_message": original_input,
            "factual_extraction": extracted,
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )


    @classmethod
    def _validate_extraction(
        cls,
        result: Any,
    ) -> str:

        if not isinstance(result, dict):
            return "Extraction result must be a JSON object."

        fields = set(result.keys())

        missing = cls.REQUIRED_EXTRACTION_FIELDS - fields

        if missing:
            return (
                "Missing extraction fields: "
                + ", ".join(sorted(missing))
            )

        extra = fields - cls.REQUIRED_EXTRACTION_FIELDS

        if extra:
            return (
                "Unexpected extraction fields: "
                + ", ".join(sorted(extra))
            )

        for field in cls.LIST_EXTRACTION_FIELDS:

            if not isinstance(result[field], list):
                return (
                    f"Field '{field}' must be a list."
                )

            if not all(
                isinstance(item, str)
                for item in result[field]
            ):
                return (
                    f"Every item in '{field}' "
                    "must be a string."
                )

        for field in ("location", "time_information"):

            if not isinstance(result[field], str):
                return (
                    f"Field '{field}' must be a string."
                )

        return ""


    @classmethod
    def _validate_interpretation(
        cls,
        result: Any,
    ) -> str:

        if not isinstance(result, dict):
            return (
                "Interpretation result must be "
                "a JSON object."
            )

        fields = set(result.keys())

        missing = cls.REQUIRED_INTERPRETATION_FIELDS - fields

        if missing:
            return (
                "Missing interpretation fields: "
                + ", ".join(sorted(missing))
            )

        extra = fields - cls.REQUIRED_INTERPRETATION_FIELDS

        if extra:
            return (
                "Unexpected interpretation fields: "
                + ", ".join(sorted(extra))
            )

        for field in (
            "situation_summary",
            "problem_type",
        ):
            if not isinstance(result[field], str):
                return f"Field '{field}' must be a string."

        for field in ("potential_issues", "retrieval_information_needs", "missing_information"):
            if not isinstance(result[field], list):
                return f"Field '{field}' must be a list."
            if not all(isinstance(item, str) for item in result[field]):
                return f"Every item in '{field}' must be a string."

        if not isinstance(
            result["missing_information"],
            list,
        ):
            return (
                "Field 'missing_information' "
                "must be a list."
            )

        if not all(
            isinstance(item, str)
            for item in result["missing_information"]
        ):
            return (
                "Every item in 'missing_information' "
                "must be a string."
            )

        return ""


    @classmethod
    def _normalize_extraction(
        cls,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:

        normalized = {}

        for field in cls.LIST_EXTRACTION_FIELDS:
            normalized[field] = cls._deduplicate(
                result[field]
            )

        normalized["location"] = (
            result["location"].strip()
        )

        normalized["time_information"] = (
            result["time_information"].strip()
        )

        return normalized


    @staticmethod
    def _normalize_interpretation(
        result: Dict[str, Any],
    ) -> Dict[str, Any]:

        return {
            "situation_summary": result["situation_summary"].strip(),
            "problem_type": result["problem_type"].strip(),
            "potential_issues": SituationAnalyzer._deduplicate(result["potential_issues"]),
            "retrieval_information_needs": SituationAnalyzer._deduplicate(result["retrieval_information_needs"]),
            "missing_information": (
                SituationAnalyzer._deduplicate(
                    result["missing_information"]
                )
            ),
        }


    @staticmethod
    def _deduplicate(items):
        """
        Generic duplicate removal.
        """

        result = []
        seen = set()

        for item in items:

            value = item.strip()

            if not value:
                continue

            key = value.casefold()

            if key in seen:
                continue

            seen.add(key)
            result.append(value)

        return result


    @staticmethod
    def _validate_grounding(
        original_input: str,
        extracted: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Basic generic grounding safeguard.

        This intentionally does NOT contain domain-specific keywords.

        It checks whether the extraction is suspiciously empty or
        structurally inconsistent. More sophisticated semantic
        claim-verification can be added as a separate component later.
        """

        if not original_input.strip():
            return {
                "valid": False,
                "error": "Original citizen input is empty.",
            }

        total_extracted_items = sum(
            len(extracted[field])
            for field in (
                "facts",
                "people_or_entities_involved",
                "actions_or_events",
                "documents_or_evidence_mentioned",
                "important_terms",
            )
        )

        if total_extracted_items == 0:
            return {
                "valid": False,
                "error": (
                    "No factual information was extracted "
                    "from the citizen's message."
                ),
            }

        return {
            "valid": True,
            "error": "",
        }


    @staticmethod
    def _clean_json(text: str) -> str:
        """
        Clean common JSON formatting mistakes from LLM output.
        """

        if not text:
            return ""

        text = text.strip()

        # Remove Markdown code fences.
        if text.startswith("```"):
            lines = text.splitlines()

            if lines:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        # Remove accidental text surrounding JSON.
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        return text.strip()