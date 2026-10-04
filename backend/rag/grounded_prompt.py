from typing import Any, Dict, List


SYSTEM_PROMPT = """
You are NyaayMitra, a grounded legal assistance system for India.

Your task is to generate a structured legal response using ONLY:
1. The user's question.
2. The retrieved verified legal evidence.

STRICT RULES:

- Do not invent laws, sections, articles, rules, rights, responsibilities,
  procedures, authorities, or facts.
- Use ONLY information supported by the retrieved evidence.
- Every legal provision mentioned must be supported by the retrieved evidence.
- Do not claim that a provision definitely applies unless the evidence
  supports that conclusion.
- When applicability depends on missing facts, use wording such as
  "may apply", "may be relevant", or "if the stated conditions are satisfied".
- Clearly mention important missing facts in the important note.
- Do not use outside legal knowledge.
- Return ONLY one valid JSON object.
- No markdown.
- No explanation outside JSON.

The response must contain:

situation
legal_area
applicable_law
rights
responsibilities
evidence_checklist
recommended_next_steps
authority
important_note
status

For applicable_law:
- Include only laws/provisions supported by retrieved evidence.
- Include the exact section/article/rule when present in the evidence.
- Explain the provision using the retrieved evidence.
- Explain why it may be relevant.
- Do not invent section numbers.

For legal_area:
- Identify the subject of the user's legal problem.
- Prefer a specific subject such as "Dowry Death" over a broad category
  such as "Police & Criminal Law".
- Do not use an Act name as the legal area.
- If a specific subject cannot safely be determined from the evidence,
  use the available broader subject.

For rights, responsibilities, evidence checklist, next steps and authority:
- Include only information grounded in the retrieved evidence.
- Do not invent unsupported rights or procedures.

The status must be exactly one of:
SUPPORTED
PARTIALLY_SUPPORTED
INSUFFICIENT_INFORMATION
NO_RELEVANT_SOURCE
"""


def build_grounded_prompt(
    query: str,
    evidence: List[Dict[str, Any]],
) -> str:

    evidence_blocks = []

    for index, item in enumerate(evidence, start=1):

        metadata = item.get("metadata", {}) or {}

        source_file = metadata.get(
            "source_file",
            "",
        )

        section_numbers = metadata.get(
            "section_numbers",
            [],
        )

        page = metadata.get(
            "page",
            "",
        )

        text = item.get(
            "text",
            "",
        )

        evidence_blocks.append(
            f"EVIDENCE {index}\n"
            f"SOURCE: {source_file}\n"
            f"SECTION: {section_numbers}\n"
            f"PAGE: {page}\n"
            f"TEXT:\n{text}"
        )

    evidence_text = "\n\n".join(
        evidence_blocks
    )

    return f"""
USER QUESTION:
{query}

VERIFIED RETRIEVED EVIDENCE:
{evidence_text}

TASK:

Generate a grounded legal response for the user's question.

Use ONLY the verified evidence above.

Return exactly ONE JSON object with this structure:

{{
  "status": "SUPPORTED",

  "situation": "",

  "legal_area": [],

  "applicable_law": [
    {{
      "law_name": "",
      "provision": "",
      "explanation": "",
      "relevance": ""
    }}
  ],

  "rights": [],

  "responsibilities": [],

  "evidence_checklist": [],

  "recommended_next_steps": [],

  "authority": [],

  "important_note": ""
}}

IMPORTANT:

1. SITUATION

Briefly describe the user's situation.

2. LEGAL AREA

Identify the specific subject of the user's question.

For example:

If the question is:
"One of my neighbour women died due to burn marks.
Is it a dowry death?"

Use:

[
  "Dowry Death"
]

Do NOT use a broad category if a specific subject is clear.

3. APPLICABLE LAW

For every provision:

- law_name = exact legal source supported by evidence
- provision = exact section/article/rule supported by evidence
- explanation = what the retrieved evidence says
- relevance = why it may be relevant

Do not invent legal provisions.

4. RIGHTS

List rights only when supported by the evidence.

5. RESPONSIBILITIES

List responsibilities of relevant persons or authorities only when
supported by the evidence.

6. EVIDENCE CHECKLIST

List documents, records, photographs, reports, messages, witnesses,
or other evidence only when supported by the evidence.

7. RECOMMENDED NEXT STEPS

Give practical next steps supported by the retrieved evidence.

8. AUTHORITY

List relevant authorities or complaint mechanisms supported by evidence.

9. IMPORTANT NOTE

Mention important missing facts or limitations.

For example, if the evidence does not establish:
- whether the deceased was married within seven years,
- whether dowry-related cruelty occurred,
- or the actual cause of death,

say so clearly.

Do not make a final legal determination when the required facts are
not established.

Return JSON only.
""".strip()