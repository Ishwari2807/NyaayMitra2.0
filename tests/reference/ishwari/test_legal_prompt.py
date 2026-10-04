from src.prompts.legal_prompt import LegalPrompt


def main():
    print("========================================")
    print("         LEGAL PROMPT TEST")
    print("========================================")

    # ---------------------------------------------------------
    # STEP 1: Create the user's question
    # ---------------------------------------------------------

    question = (
        "I bought a defective product online and "
        "the seller is refusing to give me a refund."
    )

    # ---------------------------------------------------------
    # STEP 2: Sample chunks retrieved from ChromaDB
    # ---------------------------------------------------------

    context = [
        (
            "An e-commerce entity shall not refuse to take back "
            "goods or services or refuse to refund consideration "
            "paid, where such goods or services are defective."
        ),
        (
            "A consumer has the right to seek redressal against "
            "unfair trade practices or restrictive trade practices."
        ),
        (
            "The e-commerce entity shall display information "
            "relating to return, refund, exchange and warranty "
            "policies."
        )
    ]

    # ---------------------------------------------------------
    # STEP 3: Create the prompt
    # ---------------------------------------------------------

    legal_prompt = LegalPrompt()

    prompt = legal_prompt.create_prompt(
        question=question,
        context=context
    )

    # ---------------------------------------------------------
    # STEP 4: Display the final prompt
    # ---------------------------------------------------------

    print("\nUSER QUESTION:")
    print(question)

    print("\nRETRIEVED CONTEXT:")
    for index, chunk in enumerate(context, start=1):
        print(f"\n[Source {index}]")
        print(chunk)

    print("\n========================================")
    print("          FINAL LLM PROMPT")
    print("========================================")

    print(prompt)

    # ---------------------------------------------------------
    # STEP 5: Basic checks
    # ---------------------------------------------------------

    assert question in prompt

    for chunk in context:
        assert chunk in prompt

    print("\n========================================")
    print("✓ LEGAL PROMPT TEST PASSED")
    print("========================================")


if __name__ == "__main__":
    main()