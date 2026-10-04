from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag.llm_client import OllamaClient


client = OllamaClient(
    model="qwen3:4b",
    timeout=300,
)

system_prompt = """
Extract factual information from the user's message.

Use ONLY the information explicitly stated by the user.

Do not invent facts.
Do not infer missing facts.
Do not provide legal advice.
Do not identify laws or offences.

Return ONLY valid JSON:

{
  "facts": [],
  "actions": [],
  "people": [],
  "time": "",
  "location": "",
  "evidence": []
}
"""

user_input = "My landlord is refusing to return my deposit."

print("=" * 80)
print("Direct LLM Extraction Test")
print("=" * 80)

print("\nInput:")
print(user_input)

print("\nCalling model...")

response = client.generate(
    system_prompt,
    user_input,
)

print("\nRaw model response:")
print(response)

print("\n" + "=" * 80)