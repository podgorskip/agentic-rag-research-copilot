from pydantic import Field

from ._base import ImmutableModel, MutableModel
from ._entities import Chunk, Citation


class GradeResult(ImmutableModel):
    """Used by the LLM to grade retrieved chunk relevance."""

    is_relevant: bool = Field(..., description="True if chunks contain enough info to answer")
    reasoning: str = Field(..., description="Explanation of why context is sufficient or lacking")


class SynthesizedAnswer(ImmutableModel):
    """Structured output returned by the LLM generation step."""

    answer: str = Field(..., description="The synthesized answer with inline citation markers")
    citations: list[Citation] = Field(
        default_factory=list[Citation], description="Attributed sources"
    )


class AgentState(MutableModel):
    """Tracks state across the Rewrite -> Retrieve -> Grade -> Generate loop."""

    original_query: str
    current_query: str
    query_history: list[str] = Field(
        default_factory=list[str], description="Tracks rewritten queries"
    )
    retrieved_chunks: list[Chunk] = Field(default_factory=list[Chunk])
    is_context_sufficient: bool = False
    loop_count: int = Field(default=0, ge=0, description="Prevents infinite self-correction loops")
    final_answer: SynthesizedAnswer | None = None
