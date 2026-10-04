from backend.rag.grounding_validator import GroundingValidator


def main():
    validator = GroundingValidator()

    evidence = [
        {
            "text": """
            If any person without permission of the owner accesses
            a computer system or computer network, he shall be liable
            to pay damages by way of compensation to the person so affected.
            """
        }
    ]

    # ----------------------------------------------------------
    # Test 1: supported content
    # ----------------------------------------------------------

    supported_response = {
        "status": "SUPPORTED",
        "issue_detected": "Access to a computer without permission.",
        "situation": "A person accesses a computer without permission.",
        "what_may_be_relevant": [
            "Accessing a computer system without permission"
        ],
        "what_rights_may_apply": [],
        "recommended_next_steps": [],
        "evidence_checklist": [],
        "limitations": [],
    }

    result = validator.validate(
        supported_response,
        evidence,
    )

    print("\nTEST 1 — Supported content")
    print(result)

    # ----------------------------------------------------------
    # Test 2: unsupported recommendation
    # ----------------------------------------------------------

    unsupported_response = {
        "status": "SUPPORTED",
        "issue_detected": "Access to a computer without permission.",
        "situation": "A person accesses a computer without permission.",
        "what_may_be_relevant": [
            "Accessing a computer system without permission"
        ],
        "what_rights_may_apply": [],
        "recommended_next_steps": [
            "Report the incident to the police immediately."
        ],
        "evidence_checklist": [],
        "limitations": [],
    }

    result = validator.validate(
        unsupported_response,
        evidence,
    )

    print("\nTEST 2 — Unsupported recommendation")
    print(result)


if __name__ == "__main__":
    main()