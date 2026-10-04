from typing import List
from pydantic import BaseModel, Field


class LegalResponse(BaseModel):
    """User-facing structured NyaayMitra response contract."""

    situation: str = Field(default="")
    legal_area: List[str] = Field(default_factory=list)
    rights: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    authority: List[str] = Field(default_factory=list)
    sources: List[dict] = Field(default_factory=list)
    important_note: str = Field(default="")
