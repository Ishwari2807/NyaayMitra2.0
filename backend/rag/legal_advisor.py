from typing import Any, Dict
from backend.rag.advisor_graph import AdvisorGraph


class LegalAdvisor:
    """Public end-to-end facade over the single AdvisorGraph workflow."""

    def __init__(self, retriever, reranker, evidence_gate, generator, situation_analyzer=None):
        self.graph = AdvisorGraph(
            retriever=retriever,
            reranker=reranker,
            evidence_gate=evidence_gate,
            generator=generator,
            situation_analyzer=situation_analyzer,
        )

    def answer(self, query: str, retrieval_k: int = 10, top_k: int = 5) -> Dict[str, Any]:
        return self.graph.run(query=query, retrieval_k=retrieval_k, top_k=top_k)
