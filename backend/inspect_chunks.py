import json
import glob


def inspect_file(pattern, name):
    files = glob.glob(pattern)

    if not files:
        print(f"\n{name}: file not found")
        return

    path = files[0]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    legal_chunks = [
        chunk
        for chunk in data
        if chunk.get("metadata", {}).get("chunk_type")
        == "legal_section"
    ]

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("Total chunks:", len(data))
    print("Legal section chunks:", len(legal_chunks))

    for chunk in legal_chunks[:5]:
        print("\n" + "-" * 70)
        print("Chunk ID:", chunk.get("chunk_id"))
        print("Sections:", chunk.get("metadata", {}).get("section_numbers"))
        print("Pages:", chunk.get("pages"))
        print("Structure path:", chunk.get("structure_path"))
        print("\nTEXT:")
        print(chunk.get("text", "")[:2000])


inspect_file(
    "data/chunks/bharatiya*.json",
    "BHARATIYA NYAYA SANHITA"
)

inspect_file(
    "data/chunks/information*.json",
    "INFORMATION TECHNOLOGY ACT"
)