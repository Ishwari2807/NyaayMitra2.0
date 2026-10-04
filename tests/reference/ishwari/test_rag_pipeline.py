from src.pipeline.rag_pipeline import RAGPipeline


def display_response(response):
    print("\n========================================")
    print("             CIVIC GUIDANCE")
    print("========================================")

    print("\n📌 Situation")
    print(response.situation)

    print("\n⚖️ Legal Area")
    print(response.legal_area)

    print("\n✅ Rights")
    for item in response.rights:
        print(f"• {item}")

    print("\n📋 Responsibilities")
    for item in response.responsibilities:
        print(f"• {item}")

    print("\n📄 Evidence to Keep")
    for item in response.evidence:
        print(f"• {item}")

    print("\n➡️ Next Steps")
    for index, item in enumerate(response.next_steps, start=1):
        print(f"{index}. {item}")

    print("\n🏛️ Authority / Complaint Mechanism")
    for item in response.authority:
        print(f"• {item}")

    print("\n📚 Sources")
    for item in response.sources:
        print(f"• {item}")

    

    print("\n========================================")


def main():
    print("========================================")
    print("       CIVIC GUIDES LEGAL ADVISOR")
    print("========================================")

    rag = RAGPipeline()

    print(f"\nKnowledge base chunks: {rag.chroma_store.count()}")

    print("\nEnter your legal/civic question.")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() == "exit":
            print("\nExiting CivicGuides...")
            break

        if not question:
            print("Please enter a question.")
            continue

        print("\nProcessing your question...")

        response = rag.run(question)

        if response is None:
            print("\nThis question does not appear to be related to")
            print("legal or civic matters.")
            print(
                "Please ask a question related to law, legal rights, "
                "civic rules, responsibilities, complaints, "
                "or government procedures."
            )
            continue

        display_response(response)


if __name__ == "__main__":
    main()