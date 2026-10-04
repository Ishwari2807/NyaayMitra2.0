from src.llm.groq_llm import GroqLLM


def main():
    print("========================================")
    print("          GROQ LLM TEST")
    print("========================================")

    # ---------------------------------------------------------
    # STEP 1: Create the LLM
    # ---------------------------------------------------------

    llm = GroqLLM()

    # ---------------------------------------------------------
    # STEP 2: Create a simple test prompt
    # ---------------------------------------------------------

    prompt = """
You are a civic and legal awareness assistant.

Explain in simple language what a consumer right is.
Keep the answer within 3 sentences.
"""

    # ---------------------------------------------------------
    # STEP 3: Send the prompt to Groq
    # ---------------------------------------------------------

    print("\nSending prompt to Groq...")

    response = llm.generate(prompt)

    # ---------------------------------------------------------
    # STEP 4: Display the response
    # ---------------------------------------------------------

    print("\nLLM Response:")
    print(response)

    # ---------------------------------------------------------
    # STEP 5: Basic validation
    # ---------------------------------------------------------

    assert isinstance(response, str)
    assert len(response.strip()) > 0

    print("\n========================================")
    print("✓ GROQ LLM TEST PASSED")
    print("========================================")


if __name__ == "__main__":
    main()