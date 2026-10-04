from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag.llm_client import OllamaClient


def main():
    print("=" * 80)
    print("NyaayMitra — Minimal Grounded Generation Test")
    print("=" * 80)

    client = OllamaClient(
        model="llama3.2:1b",
        timeout=300,
    )

    query = "What happens if someone gets access to my computer without permission?"

    evidence = """43. [Penalty and compensation] for damage to computer, computer system, etc.–If any person
without permission of the owner or any other person who is in charge of a computer, computer system or
computer network,–"""

    system_prompt = """You are a legal information extraction system.

Use ONLY the evidence provided below.

Do not use your own knowledge.
Do not invent facts.
Do not invent legal rules.
Do not infer information that is not stated in the evidence.

Return ONLY valid JSON with exactly these fields:

{
  "relevant": true,
  "section": "",
  "evidence_summary": ""
}

The "section" field must contain the section number explicitly present in the evidence.
The "evidence_summary" must only summarize what is explicitly stated in the evidence.

If the evidence does not answer the question, set "relevant" to false.

Return JSON only.
"""

    user_prompt = f"""USER QUESTION:
{query}

EVIDENCE:
{evidence}
"""

    print("\nSending minimal task to Llama 3.2 1B...")
    print("-" * 80)

    raw_response = client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    print("\nRAW MODEL RESPONSE:")
    print("-" * 80)
    print(raw_response)

    print("\nPARSED JSON:")
    print("-" * 80)

    try:
        parsed = json.loads(raw_response)
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print("MODEL DID NOT RETURN VALID JSON")

    print("\n" + "=" * 80)
    print("Test complete")
    print("=" * 80)


if __name__ == "__main__":
    main()