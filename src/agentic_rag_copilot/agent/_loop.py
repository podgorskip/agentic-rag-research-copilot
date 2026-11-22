import time

from langfuse import Langfuse, get_client

from agentic_rag_copilot.domain.models import (
    AgentState,
    AskRequest,
    AskResponse,
    Chunk,
    GradeResult,
    Query,
    SynthesizedAnswer,
)
from agentic_rag_copilot.logging import get_logger
from agentic_rag_copilot.services import RetrievalService

from ._grader import RelevanceGrader
from ._rewriter import QueryRewriter
from ._settings import AgentSettings
from ._synthesizer import AnswerSynthesizer

_logger = get_logger(__name__)


class AgenticLoopService:
    """Self-correcting Rewrite -> Retrieve -> Grade -> Generate loop over ``RetrievalService``.

    Each pass retrieves chunks for ``state.current_query`` and merges them into
    ``state.retrieved_chunks`` (de-duplicated by ``Chunk.id``), so later hops add to earlier
    context. The loop stops when the grader reports sufficient context or after
    ``settings.max_loop_count`` rewrites, then always synthesizes a best-effort answer.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        grader: RelevanceGrader,
        rewriter: QueryRewriter,
        synthesizer: AnswerSynthesizer,
        settings: AgentSettings,
        langfuse_client: Langfuse | None = None,
    ) -> None:
        """Initialize the agentic loop service.

        Args:
            retrieval_service: Two-stage retriever invoked once per loop pass.
            grader: Decides whether accumulated chunks suffice for the current query.
            rewriter: Reformulates the query when context is insufficient.
            synthesizer: Produces the final cited answer.
            settings: Agent configuration; ``max_loop_count`` caps the number of rewrites.
            langfuse_client: Langfuse client used to emit step spans. Defaults to the
                process-wide client from ``langfuse.get_client()``.
        """
        self._retrieval = retrieval_service
        self._grader = grader
        self._rewriter = rewriter
        self._synthesizer = synthesizer
        self._max_loop_count = settings.max_loop_count
        self._langfuse = langfuse_client if langfuse_client is not None else get_client()

    async def ask(self, request: AskRequest) -> AskResponse:
        """Answer a question by iterating retrieval, grading and rewriting until sufficient.

        Args:
            request: User question, final ``top_k`` per retrieval pass, and metadata filters.

        Returns:
            Answer with grounded citations and the number of retrieve/grade passes performed.
            When ``max_loop_count`` is exhausted without sufficient context, the answer is a
            best-effort synthesis over everything retrieved so far.

        Raises:
            LLMError: Propagated if an LLM call fails.
            ValueError: Propagated if the rewriter returns an empty query or the synthesizer
                rejects a citation of an unknown chunk.
        """
        state = AgentState(
            original_query=request.query,
            current_query=request.query,
            query_history=[request.query],
        )
        iterations = 0

        with self._langfuse.start_as_current_span(
            name="agentic_loop",
            input={"query": request.query, "top_k": request.top_k},
        ) as span:
            while True:
                retrieved = await self._retrieval.retrieve(
                    Query(text=state.current_query, filters=request.filters),
                    final_top_k=request.top_k,
                )
                iterations += 1
                added = _merge_chunks(state, retrieved)
                _logger.info(
                    f"Pass {iterations}: query={state.current_query!r} retrieved={len(retrieved)} "
                    f"new={added} total={len(state.retrieved_chunks)}"
                )

                grade = await self._traced_grade(state, iterations)
                state.is_context_sufficient = grade.is_relevant
                if state.is_context_sufficient or state.loop_count >= self._max_loop_count:
                    break

                state.current_query = await self._traced_rewrite(state, grade.reasoning)
                state.query_history.append(state.current_query)
                state.loop_count += 1

            if not state.is_context_sufficient:
                _logger.warning(
                    f"max_loop_count={self._max_loop_count} reached without sufficient context; "
                    "synthesizing best-effort answer."
                )

            state.final_answer = await self._traced_generate(state)
            span.update(
                output={"iterations_taken": iterations, "answer": state.final_answer.answer},
                metadata={
                    "iterations_taken": iterations,
                    "rewrites": state.loop_count,
                    "is_context_sufficient": state.is_context_sufficient,
                    "query_history": list(state.query_history),
                    "chunk_count": len(state.retrieved_chunks),
                },
            )

        return AskResponse(
            query=state.original_query,
            answer=state.final_answer.answer,
            citations=list(state.final_answer.citations),
            iterations_taken=iterations,
        )

    async def _traced_grade(self, state: AgentState, iteration: int) -> GradeResult:
        """Grade the accumulated context inside a Langfuse span.

        Args:
            state: Current agent state.
            iteration: 1-based retrieve/grade pass number.

        Returns:
            The grader's structured result.
        """
        with self._langfuse.start_as_current_span(
            name="grade",
            input={"query": state.current_query, "chunk_count": len(state.retrieved_chunks)},
        ) as span:
            start = time.perf_counter()
            result = await self._grader.grade(state)
            latency_ms = (time.perf_counter() - start) * 1000
            span.update(
                output={"is_relevant": result.is_relevant, "reasoning": result.reasoning},
                metadata={
                    "latency_ms": latency_ms,
                    "iteration": iteration,
                    "chunk_count": len(state.retrieved_chunks),
                    "is_relevant": result.is_relevant,
                },
            )
        return result

    async def _traced_rewrite(self, state: AgentState, reasoning: str) -> str:
        """Rewrite the current query inside a Langfuse span.

        Args:
            state: Current agent state.
            reasoning: Grader explanation of why the context was insufficient.

        Returns:
            The rewritten query text.
        """
        with self._langfuse.start_as_current_span(
            name="rewrite",
            input={"current_query": state.current_query, "reasoning": reasoning},
        ) as span:
            start = time.perf_counter()
            rewritten = await self._rewriter.rewrite(state, reasoning=reasoning)
            latency_ms = (time.perf_counter() - start) * 1000
            is_repeat = rewritten in state.query_history
            if is_repeat:
                _logger.warning(f"Rewriter repeated a previous query: {rewritten!r}")
            span.update(
                output={"rewritten_query": rewritten},
                metadata={
                    "latency_ms": latency_ms,
                    "loop_count": state.loop_count,
                    "is_repeat": is_repeat,
                },
            )
        return rewritten

    async def _traced_generate(self, state: AgentState) -> SynthesizedAnswer:
        """Synthesize the final answer inside a Langfuse span.

        Args:
            state: Final agent state with accumulated chunks.

        Returns:
            The synthesized answer with grounded citations.
        """
        with self._langfuse.start_as_current_span(
            name="generate",
            input={
                "query": state.original_query,
                "chunk_count": len(state.retrieved_chunks),
                "is_context_sufficient": state.is_context_sufficient,
            },
        ) as span:
            start = time.perf_counter()
            answer = await self._synthesizer.synthesize(state)
            latency_ms = (time.perf_counter() - start) * 1000
            span.update(
                output={"answer": answer.answer, "citation_count": len(answer.citations)},
                metadata={"latency_ms": latency_ms, "citation_count": len(answer.citations)},
            )
        return answer


def _merge_chunks(state: AgentState, new_chunks: list[Chunk]) -> int:
    """Append chunks not already in ``state.retrieved_chunks``, de-duplicating by ``Chunk.id``.

    Earlier hops keep their position; the first occurrence of a chunk wins.

    Args:
        state: Agent state whose ``retrieved_chunks`` is extended in place.
        new_chunks: Chunks returned by the latest retrieval pass.

    Returns:
        Number of chunks actually added.
    """
    seen = {chunk.id for chunk in state.retrieved_chunks}
    added = 0
    for chunk in new_chunks:
        if chunk.id not in seen:
            seen.add(chunk.id)
            state.retrieved_chunks.append(chunk)
            added += 1
    return added
