"""Optional model-backed test.

Prerequisites:
1. Put real chunk JSON files in data/chunks/ and build the Chroma index:
       python -m backend.rag.build_index
2. Install/run Ollama and pull the configured model (default llama3.2:1b).
3. Run:
       python live_test.py
"""
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
from backend.rag.llm_client import OllamaClient
from backend.rag.situation_analyzer import SituationAnalyzer
from backend.rag.hybrid_retriever import HybridRetriever
from backend.rag.reranker import LegalReranker
from backend.rag.evidence_gate import EvidenceGate
from backend.rag.grounded_generator import GroundedAnswerGenerator
from backend.rag.legal_advisor import LegalAdvisor


def main():
    model = os.getenv("NYAAYMITRA_OLLAMA_MODEL", "llama3.2:1b")
    llm = OllamaClient(model=model)
    advisor = LegalAdvisor(
        retriever=HybridRetriever(),
        reranker=LegalReranker(),
        evidence_gate=EvidenceGate(),
        generator=GroundedAnswerGenerator(llm_generate=llm.generate),
        situation_analyzer=SituationAnalyzer(llm_generate=llm.generate),
    )
    query = "I accepted a contract online and the other party says it is invalid simply because it was electronic. What does the law say?"
    print(advisor.answer(query))


if __name__ == "__main__":
    main()
