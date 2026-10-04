from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag.llm_client import OllamaClient
from backend.rag.grounded_prompt import SYSTEM_PROMPT, build_grounded_prompt


def main():
    print("=" * 80)
    print("NyaayMitra — Direct Grounded Generation Test")
    print("=" * 80)

    client = OllamaClient(
        model="llama3.2:1b",
        timeout=300,
    )

    query = "What happens if someone gets access to my computer without permission?"

    evidence = [
        {
            "chunk_id": "information_technology_act__2000__chunk_00366",
            "text": """43. [Penalty and compensation] for damage to computer, computer system, etc.–If any person
without permission of the owner or any other person who is in charge of a computer, computer system or
computer network,–""",
            "metadata": {
                "source_file": "information_technology_act__2000",
                "source_authority": "India Code",
                "document_id": "information_technology_act__2000",
                "section_numbers": "43",
                "pages": "20",
                "structure_path": "CHAPTER IX > Section 43",
            },
        }
    ]

    prompt = build_grounded_prompt(
        query=query,
        evidence=evidence,
    )

    print("\nSending directly to Llama 3.2 1B...")
    print("-" * 80)

    raw_response = client.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
    )

    print("\nRAW MODEL RESPONSE:")
    print("-" * 80)
    print(raw_response)

    print("\n" + "=" * 80)
    print("Test complete")
    print("=" * 80)


if __name__ == "__main__":
    main()