import json
import sys
import time
from pathlib import Path

# ------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.rag.llm_client import OllamaClient
from backend.rag.situation_analyzer import SituationAnalyzer


def main():

    print("=" * 80)
    print("NyaayMitra — Situation Analyzer Diagnostic Test")
    print("=" * 80)

    client = OllamaClient(
        model="llama3.2:1b",
        timeout=300,
    )

    analyzer = SituationAnalyzer(
        llm_generate=client.generate
    )

    # --------------------------------------------------------
    # Test only ONE situation first
    # --------------------------------------------------------

    user_input = "My landlord is refusing to return my deposit."

    print("\nModel:")
    print(client.model)

    print("\nTimeout:")
    print(client.timeout, "seconds")

    print("\nCitizen input:")
    print(user_input)

    print("\nStarting analysis...")

    start_time = time.perf_counter()

    try:

        result = analyzer.analyze(user_input)

        elapsed = time.perf_counter() - start_time

        print("\nResponse received.")

        print(
            f"Elapsed time: {elapsed:.2f} seconds"
        )

        print("\nResult:")

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as exc:

        elapsed = time.perf_counter() - start_time

        print("\nERROR")

        print(
            f"Elapsed time before error: "
            f"{elapsed:.2f} seconds"
        )

        print(
            "Exception type:",
            type(exc).__name__,
        )

        print(
            "Exception:",
            str(exc),
        )

    print("\n" + "=" * 80)
    print("Diagnostic test complete")
    print("=" * 80)


if __name__ == "__main__":
    main()