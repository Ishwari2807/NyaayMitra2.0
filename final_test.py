import io
import json
from contextlib import redirect_stdout, redirect_stderr

from backend.rag.advisor_pipeline import AdvisorPipeline


def has_content(value):
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, list):
        return len(value) > 0

    if isinstance(value, dict):
        return len(value) > 0

    return True


def print_bullets(items):

    if not isinstance(items, list):
        return

    for item in items:

        if isinstance(item, dict):

            for key, value in item.items():
                if has_content(value):
                    print(f"• {key}: {value}")

        elif has_content(item):
            print(f"• {item}")


def print_numbered(items):

    if not isinstance(items, list):
        return

    number = 1

    for item in items:

        if not has_content(item):
            continue

        if isinstance(item, str):

            # Prevent "1. 1. Contact..."
            cleaned = item.strip()

            if cleaned[:3].isdigit():
                parts = cleaned.split(".", 1)

                if len(parts) == 2:
                    cleaned = parts[1].strip()

            print(f"{number}. {cleaned}")

        else:
            print(f"{number}. {item}")

        number += 1


def display_response(response):

    print("\n" + "=" * 70)
    print("                         LEGAL GUIDANCE")
    print("=" * 70)

    # ---------------------------------------------------------
    # Situation
    # ---------------------------------------------------------

    if has_content(
        response.get("situation")
    ):

        print("\n📌 Situation")
        print(response["situation"])

    # ---------------------------------------------------------
    # Legal Area
    # ---------------------------------------------------------

    legal_area = response.get(
        "legal_area",
        []
    )

    if has_content(legal_area):

        print("\n⚖️ Legal Area")
        print_bullets(legal_area)

    # ---------------------------------------------------------
    # Applicable Law
    # ---------------------------------------------------------

    applicable_law = response.get(
        "applicable_law",
        []
    )

    if has_content(applicable_law):

        print("\n📚 Applicable Law")

        for law in applicable_law:

            if not isinstance(law, dict):
                print(f"• {law}")
                continue

            law_name = law.get(
                "law_name",
                "Legal provision"
            )

            provision = law.get(
                "provision",
                "Not specified in retrieved evidence"
            )

            explanation = law.get(
                "explanation",
                ""
            )

            relevance = law.get(
                "relevance",
                ""
            )

            print(f"• {law_name}")

            if has_content(provision):
                print(
                    f"  Section / Article / Rule: {provision}"
                )

            if has_content(explanation):
                print(
                    f"  Explanation: {explanation}"
                )

            if has_content(relevance):
                print(
                    f"  Relevance: {relevance}"
                )

            print()

    # ---------------------------------------------------------
    # Rights
    # ---------------------------------------------------------

    rights = response.get(
        "rights",
        response.get(
            "what_rights_may_apply",
            []
        )
    )

    if has_content(rights):

        print("✅ Rights")
        print_bullets(rights)

    # ---------------------------------------------------------
    # Responsibilities
    # ---------------------------------------------------------

    responsibilities = response.get(
        "responsibilities",
        []
    )

    if has_content(responsibilities):

        print("\n📋 Responsibilities")
        print_bullets(responsibilities)

    # ---------------------------------------------------------
    # Evidence to Keep
    # ---------------------------------------------------------

    evidence = response.get(
        "evidence_checklist",
        []
    )

    if not has_content(evidence):

        evidence = response.get(
            "evidence",
            []
        )

    if has_content(evidence):

        print("\n📄 Evidence to Keep")
        print_bullets(evidence)

    # ---------------------------------------------------------
    # Next Steps
    # ---------------------------------------------------------

    next_steps = response.get(
        "recommended_next_steps",
        []
    )

    if not has_content(next_steps):

        next_steps = response.get(
            "next_steps",
            []
        )

    if has_content(next_steps):

        print("\n➡️ Next Steps")
        print_numbered(next_steps)

    # ---------------------------------------------------------
    # Authority / Complaint Mechanism
    # ---------------------------------------------------------

    authority = response.get(
        "authority",
        []
    )

    if has_content(authority):

        print(
            "\n🏛️ Authority / Complaint Mechanism"
        )

        print_bullets(authority)

    # ---------------------------------------------------------
    # Sources
    # ---------------------------------------------------------

    sources = response.get(
        "sources",
        []
    )

    if has_content(sources):

        print("\n📚 Sources")

        source_number = 1

        for source in sources:

            if not isinstance(
                source,
                dict,
            ):
                print(
                    f"• Source {source_number}: {source}"
                )
                source_number += 1
                continue

            source_file = source.get(
                "source_file",
                ""
            )

            section = source.get(
                "section_numbers",
                ""
            )

            page = source.get(
                "page",
                ""
            )

            text = source_file

            if has_content(section):
                text += f" | Section: {section}"

            if has_content(page):
                text += f" | Page: {page}"

            if has_content(text):

                print(
                    f"• Source {source_number}: {text}"
                )

                source_number += 1

    # ---------------------------------------------------------
    # Important Note
    # ---------------------------------------------------------

    important_note = response.get(
        "important_note",
        ""
    )

    if not has_content(important_note):

        limitations = response.get(
            "limitations",
            []
        )

        if isinstance(
            limitations,
            list,
        ):
            important_note = " ".join(
                str(x)
                for x in limitations
                if has_content(x)
            )

    if has_content(important_note):

        print("\n⚠️ Important Note")
        print(important_note)

    print("\n" + "=" * 70)


def quiet_create_pipeline():

    # Hide model-loading / embedding / reranker messages.
    # The pipeline itself is unchanged.
    hidden_output = io.StringIO()

    with redirect_stdout(hidden_output), redirect_stderr(hidden_output):
        pipeline = AdvisorPipeline()

    return pipeline


def quiet_analyze(
    pipeline,
    query,
):

    # Hide embedding batch/progress messages.
    hidden_output = io.StringIO()

    with redirect_stdout(hidden_output), redirect_stderr(hidden_output):
        result = pipeline.analyze(query)

    return result


def main():

    print("=" * 70)
    print("                    NYAAYMITRA")
    print("          RAG-Based Legal Assistance System")
    print("=" * 70)

    try:

        pipeline = quiet_create_pipeline()

    except Exception as exc:

        print("\nFailed to initialize NyaayMitra.")
        print(str(exc))
        return

    while True:

        query = input(
            "\nYour question: "
        ).strip()

        if query.lower() in {
            "exit",
            "quit",
        }:

            print("\nExiting NyaayMitra...")
            break

        if not query:

            print(
                "\nPlease enter a valid question."
            )

            continue

        print(
            "\nProcessing your question..."
        )

        print(
            "Please wait...\n"
        )

        try:

            result = quiet_analyze(
                pipeline,
                query,
            )

            if result is None:

                print(
                    "No grounded legal response "
                    "could be generated."
                )

                continue

            display_response(
                result
            )

        except Exception as exc:

            print(
                "\n" + "=" * 70
            )

            print("ERROR")

            print(
                "=" * 70
            )

            print(
                str(exc)
            )


if __name__ == "__main__":
    main()