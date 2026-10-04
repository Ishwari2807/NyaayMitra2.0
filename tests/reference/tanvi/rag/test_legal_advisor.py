from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.rag.hybrid_retriever import HybridRetriever
from backend.rag.reranker import LegalReranker
from backend.rag.evidence_gate import EvidenceGate
from backend.rag.grounded_generator import GroundedAnswerGenerator
from backend.rag.langchain_llm_client import LangChainLLMClient
from backend.rag.legal_advisor import LegalAdvisor


def print_section(title):
    print("\n" + title)
    print("-" * 80)


def print_list(items):
    if not items:
        print("No specific information identified from the retrieved sources.")
        return

    for item in items:
        if isinstance(item, str):
            print(f"• {item}")
        elif isinstance(item, dict):
            # Keep terminal output readable if the backend returns structured data.
            source = item.get("source")
            text = item.get("text")

            if source:
                print(f"• {source}")

            if text:
                print(f"  {text}")
        else:
            print(f"• {item}")


def print_evidence_checklist(items):
    if not items:
        print("No specific evidence identified from the retrieved sources.")
        return

    for item in items:
        if isinstance(item, str):
            print(f"☐ {item}")

        elif isinstance(item, dict):
            source = item.get("source")
            text = item.get("text")

            if source:
                print(f"☐ {source}")

            if text:
                print(f"  {text}")

        else:
            print(f"☐ {item}")


def print_sources(sources):
    if not sources:
        print("No sources returned.")
        return

    for source in sources:
        if not isinstance(source, dict):
            print(f"• {source}")
            continue

        source_file = source.get("source_file", "Unknown source")
        chunk_id = source.get("chunk_id", "Unknown chunk")
        sections = source.get("section_numbers", "")

        print(f"• {source_file}")

        if sections:
            print(f"  Section: {sections}")

        print(f"  Chunk: {chunk_id}")


def display_result(result):

    print("\n")
    print("=" * 80)
    print("                         YOUR CIVIC ADVISOR")
    print("=" * 80)

    status = result.get("status", "UNKNOWN")

    print_section("Issue detected:")
    issue = result.get("issue_detected", "")
    print(issue if issue else "Not determined from the retrieved evidence.")

    print_section("Your situation:")
    situation = result.get("situation", "")
    print(situation if situation else "Not determined from the retrieved evidence.")

    print_section("🔎 WHAT MAY BE RELEVANT")

    print("\nPotentially relevant areas:")
    print_list(result.get("what_may_be_relevant", []))

    print("\n⚖️ WHAT RIGHTS MAY APPLY")
    print_list(result.get("what_rights_may_apply", []))

    print_section("📖 APPLICABLE LAW / LEGAL EXPLANATION")
    print_evidence_checklist(result.get("evidence_checklist", []))

    print_section("🚶 RECOMMENDED NEXT STEPS")
    print_list(result.get("recommended_next_steps", []))

    print_section("📚 SOURCES")
    print_sources(result.get("sources", []))

    limitations = result.get("limitations", [])

    if limitations:
        print_section("⚠️ LIMITATIONS")
        print_list(limitations)

    grounding_warnings = result.get("grounding_warnings", [])

    if grounding_warnings:
        print_section("⚠️ GROUNDING WARNINGS")

        for warning in grounding_warnings:
            if isinstance(warning, dict):
                field = warning.get("field", "Unknown field")
                reason = warning.get("reason", "")

                print(f"• {field}")
                if reason:
                    print(f"  {reason}")
            else:
                print(f"• {warning}")

    print_section(f"Status: {status}")

    print("=" * 80)


def main():

    print("=" * 80)
    print("NyaayMitra — End-to-End Legal Advisor Test")
    print("=" * 80)

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------

    client = LangChainLLMClient(
        model="llama3.2:1b",
    )

    # --------------------------------------------------
    # Components
    # --------------------------------------------------

    retriever = HybridRetriever()

    reranker = LegalReranker()

    evidence_gate = EvidenceGate()

    generator = GroundedAnswerGenerator(
        llm_generate=client.generate,
    )

    advisor = LegalAdvisor(
        retriever=retriever,
        reranker=reranker,
        evidence_gate=evidence_gate,
        generator=generator,
    )

    # --------------------------------------------------
    # Test query
    # --------------------------------------------------

    query = (
        "I noticed an e-wallet transaction that I did not make, and money has been deducted from my wallet. How can I report this incident and what transaction information can I provide as evidence?"
    )

    print("\nCitizen query:")
    print(query)

    print("\nRunning Legal Advisor...")
    print("-" * 80)

    result = advisor.answer(
        query=query,
        retrieval_k=10,
        top_k=5,
    )

    # --------------------------------------------------
    # Human-readable output
    # --------------------------------------------------

    display_result(result)

    print("\nTest complete.")


if __name__ == "__main__":
    main()