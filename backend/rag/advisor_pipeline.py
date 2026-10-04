"""Backward-compatible facade.

The old AdvisorPipeline duplicated the LangGraph workflow. It now delegates to
LegalAdvisor so there is only one production orchestration path.
"""

import os
from typing import Any, Dict
from backend.rag.llm_client import GroqClient
from backend.rag.situation_analyzer import SituationAnalyzer
from backend.rag.hybrid_retriever import HybridRetriever
from backend.rag.reranker import LegalReranker
from backend.rag.evidence_gate import EvidenceGate
from backend.rag.grounded_generator import GroundedAnswerGenerator
from backend.rag.legal_advisor import LegalAdvisor


class AdvisorPipeline:
    """Compatibility wrapper; use LegalAdvisor/AdvisorGraph for new code."""

    def __init__(self):
        model = "openai/gpt-oss-20b"
        llm = GroqClient(model=model)
        situation_analyzer = SituationAnalyzer(llm_generate=llm.generate)
        advisor = LegalAdvisor(
            retriever=HybridRetriever(),
            reranker=LegalReranker(),
            evidence_gate=EvidenceGate(),
            generator=GroundedAnswerGenerator(llm_generate=llm.generate),
            situation_analyzer=situation_analyzer,
        )
        self.advisor = advisor

    def analyze(self, user_input: str) -> Dict[str, Any]:
        return self.advisor.answer(user_input)
