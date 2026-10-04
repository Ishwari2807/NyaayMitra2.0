import re


def classify_document(
    title: str,
    text: str,
) -> dict:

    combined = (
        f"{title}\n{text}"
    ).lower()

    legislation_terms = [
        "act,",
        "act ",
        "code",
        "sanhita",
        "ordinance",
        "statute",
    ]

    judgment_terms = [
        "judgment",
        "judgement",
        "court",
        "petition",
        "appellant",
        "respondent",
    ]

    guideline_terms = [
        "guidelines",
        "guideline",
        "advisory",
        "instructions",
        "do's and don'ts",
    ]

    faq_terms = [
        "frequently asked questions",
        "faq",
        "question and answer",
    ]

    scheme_terms = [
        "scheme",
        "eligibility",
        "benefits",
        "application process",
    ]

    if any(
        term in combined
        for term in legislation_terms
    ):
        document_type = "legislation"

    elif any(
        term in combined
        for term in judgment_terms
    ):
        document_type = "court_judgment"

    elif any(
        term in combined
        for term in guideline_terms
    ):
        document_type = "government_guideline"

    elif any(
        term in combined
        for term in faq_terms
    ):
        document_type = "faq"

    elif any(
        term in combined
        for term in scheme_terms
    ):
        document_type = "government_scheme"

    else:
        document_type = "unknown"

    return {
        "document_type": document_type,
        "classification_method": "heuristic",
    }