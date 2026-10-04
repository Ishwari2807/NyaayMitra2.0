from __future__ import annotations

from typing import Any, Dict, TypedDict
from langgraph.graph import END, START, StateGraph


class AdvisorState(TypedDict, total=False):
    query: str
    retrieval_query: str
    retrieval_k: int
    top_k: int
    situation: dict[str, Any]
    retrieved: list[dict[str, Any]]
    reranked: list[dict[str, Any]]
    evidence_result: dict[str, Any]
    answer: dict[str, Any]


class AdvisorGraph:
    """Single production workflow for NyaayMitra.

    LangGraph owns state/branching. Retrieval, RRF, reranking and grounding
    remain ordinary Python components.
    """

    def __init__(self, retriever, reranker, evidence_gate, generator, situation_analyzer=None):
        self.retriever = retriever
        self.reranker = reranker
        self.evidence_gate = evidence_gate
        self.generator = generator
        self.situation_analyzer = situation_analyzer

        graph = StateGraph(AdvisorState)
        graph.add_node("situation", self._situation_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("rerank", self._rerank_node)
        graph.add_node("evidence_gate", self._evidence_gate_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("blocked", self._blocked_node)

        graph.add_edge(START, "situation")
        graph.add_edge("situation", "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "evidence_gate")
        graph.add_conditional_edges(
            "evidence_gate",
            self._route_after_evidence_gate,
            {"generate": "generate", "blocked": "blocked"},
        )
        graph.add_edge("generate", END)
        graph.add_edge("blocked", END)
        self.graph = graph.compile()

    def _situation_node(self, state: AdvisorState) -> Dict[str, Any]:
        query = state["query"]
        if self.situation_analyzer is None:
            return {"situation": {}, "retrieval_query": query}
        situation = self.situation_analyzer.analyze(query)
        if situation.get("status") != "OK":
            return {"situation": situation, "retrieval_query": query}
        parts = [
            query,
            situation.get("situation_summary", ""),
            situation.get("problem_type", ""),
            " ".join(situation.get("potential_issues", [])),
            " ".join(situation.get("retrieval_information_needs", [])),
            " ".join(situation.get("important_terms", [])),
            " ".join(situation.get("actions_or_events", [])),
        ]
        retrieval_query = " ".join(p.strip() for p in parts if p and p.strip())
        return {"situation": situation, "retrieval_query": retrieval_query}

    def _retrieve_node(self, state: AdvisorState) -> Dict[str, Any]:
        query = state.get("retrieval_query", state["query"])
        k = state.get("retrieval_k", 10)
        return {"retrieved": self.retriever.search(query=query, top_k=k, retrieval_k=k)}

    def _rerank_node(self, state: AdvisorState) -> Dict[str, Any]:
        retrieved = state.get("retrieved", [])
        if not retrieved:
            return {"reranked": []}
        query = state.get("retrieval_query", state["query"])
        return {"reranked": self.reranker.rerank(query=query, results=retrieved, top_k=state.get("top_k", 5))}

    def _evidence_gate_node(self, state: AdvisorState) -> Dict[str, Any]:
        reranked = state.get("reranked", [])
        if not reranked:
            return {"evidence_result": {"proceed": False, "status": "NO_RELEVANT_SOURCE", "reason": "No sufficiently relevant verified source was found.", "evidence": []}}
        situation = state.get("situation", {})
        result = self.evidence_gate.evaluate(
            query=state.get("retrieval_query", state["query"]),
            results=reranked,
            situation=situation,
            information_needs=situation.get("retrieval_information_needs", []) if isinstance(situation, dict) else [],
        )
        return {"evidence_result": result}

    def _route_after_evidence_gate(self, state: AdvisorState) -> str:
        return "generate" if state.get("evidence_result", {}).get("proceed", False) else "blocked"

    def _generate_node(self, state: AdvisorState) -> Dict[str, Any]:
        evidence = state.get("evidence_result", {}).get("evidence", [])
        answer = self.generator.generate(query=state["query"], evidence=evidence)
        return {"answer": answer}

    def _blocked_node(self, state: AdvisorState) -> Dict[str, Any]:
        gate = state.get("evidence_result", {})
        return {"answer": {
            "status": gate.get("status", "INSUFFICIENT_INFORMATION"),
            "query": state["query"],
            "error": gate.get("reason", "Insufficient verified evidence."),
            "sources": self._extract_sources(gate.get("evidence", [])),
            "coverage": gate.get("coverage", {}),
            "issues_covered": gate.get("issues_covered", []),
            "issues_uncovered": gate.get("issues_uncovered", []),
        }}

    @staticmethod
    def _extract_sources(results):
        sources = []
        for result in results or []:
            metadata = result.get("metadata", {}) or {}
            sources.append({
                "chunk_id": result.get("chunk_id"),
                "source_file": metadata.get("source_file"),
                "source_authority": metadata.get("source_authority") or metadata.get("authority"),
                "document_id": metadata.get("document_id"),
                "section_numbers": metadata.get("section_numbers"),
                "pages": metadata.get("pages"),
                "document_version": metadata.get("document_version"),
                "official_source": metadata.get("official_source"),
            })
        return sources

    def run(self, query: str, retrieval_k: int = 10, top_k: int = 5) -> Dict[str, Any]:
        if not query or not query.strip():
            return {"status": "INVALID_INPUT", "error": "Please describe your situation."}
        result = self.graph.invoke({"query": query.strip(), "retrieval_k": retrieval_k, "top_k": top_k})
        return result.get("answer", {"status": "GENERATION_ERROR", "query": query.strip(), "error": "No answer was produced.", "sources": []})
