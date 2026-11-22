from typing import Final

from agentic_rag_copilot.domain.models import AgentState, Chunk, Citation, SynthesizedAnswer
from agentic_rag_copilot.domain.protocols import LLMProtocol
from agentic_rag_copilot.logging import get_logger

_logger = get_logger(__name__)

_NO_CONTEXT_ANSWER: Final[str] = (
    "I could not find any information in the ingested sources to answer this question."
)

_SYSTEM_PROMPT: Final[str] = (
    "You are a research assistant answering questions strictly from the provided context "
    "chunks. Each chunk is labelled with a bracketed number such as [1] and a chunk_id. "
    "Write a concise answer that cites its supporting chunks inline using those bracketed "
    "numbers, e.g. 'RAG uses a BERT-based retriever [1].' "
    "For every chunk you cite, add one entry to citations with: chunk_id copied exactly from "
    "the chunk label, document_id and source_uri copied exactly from the chunk label, and "
    "quote set to a short verbatim span from that chunk's text that supports the claim. "
    "Never cite a chunk_id that is not listed in the context. If the context does not "
    "contain the answer, say so plainly instead of guessing."
)


class AnswerSynthesizer:
    """Generates the final cited answer from the accumulated retrieved context.

    Citations returned by the LLM are validated against ``state.retrieved_chunks``: an unknown
    ``chunk_id`` raises ``ValueError``, and each citation's ``document_id`` / ``source_uri`` are
    overwritten with the values stored on the cited chunk so they can never be hallucinated.
    """

    def __init__(self, llm: LLMProtocol) -> None:
        """Initialize the answer synthesizer.

        Args:
            llm: LLM adapter implementing ``LLMProtocol`` for structured generation.
        """
        self._llm = llm

    async def synthesize(self, state: AgentState) -> SynthesizedAnswer:
        """Synthesize a cited answer to ``state.original_query`` from ``state.retrieved_chunks``.

        Short-circuits without calling the LLM when ``state.retrieved_chunks`` is empty,
        returning an answer that states no supporting information was found. Does not mutate
        ``state``; callers own assigning ``state.final_answer``.

        Args:
            state: Agent state providing ``original_query`` and ``retrieved_chunks``.

        Returns:
            ``SynthesizedAnswer`` whose citations all reference chunks in ``retrieved_chunks``.

        Raises:
            ValueError: If the LLM cites a ``chunk_id`` not present in ``retrieved_chunks``.
            LLMError: Propagated if the underlying LLM call fails or returns invalid schema.
        """
        if not state.retrieved_chunks:
            _logger.debug("Short-circuiting answer synthesis: no retrieved chunks.")
            return SynthesizedAnswer(answer=_NO_CONTEXT_ANSWER, citations=[])

        _logger.debug(
            f"Synthesizing answer from {len(state.retrieved_chunks)} chunk(s) for query: "
            f"{state.original_query!r}"
        )
        raw = await self._llm.generate_structured(
            _build_prompt(state),
            SynthesizedAnswer,
            system_prompt=_SYSTEM_PROMPT,
        )
        chunks_by_id = {chunk.id: chunk for chunk in state.retrieved_chunks}
        citations = [_ground_citation(citation, chunks_by_id) for citation in raw.citations]
        return SynthesizedAnswer(answer=raw.answer, citations=citations)


def _source_uri(chunk: Chunk) -> str:
    """Return the chunk's source URI from metadata, falling back to its document ID.

    Args:
        chunk: Retrieved chunk.

    Returns:
        ``chunk.metadata["source_uri"]`` if it is a non-empty string, else ``chunk.document_id``.
    """
    value = chunk.metadata.get("source_uri")
    return value if isinstance(value, str) and value else chunk.document_id


def _ground_citation(citation: Citation, chunks_by_id: dict[str, Chunk]) -> Citation:
    """Validate a citation against the retrieved chunks and copy grounded identifiers.

    Args:
        citation: Citation emitted by the LLM.
        chunks_by_id: Retrieved chunks keyed by ``Chunk.id``.

    Returns:
        Citation with ``document_id`` and ``source_uri`` taken from the cited chunk.

    Raises:
        ValueError: If ``citation.chunk_id`` does not match any retrieved chunk.
    """
    chunk = chunks_by_id.get(citation.chunk_id)
    if chunk is None:
        raise ValueError(
            f"LLM cited unknown chunk_id {citation.chunk_id!r}; "
            f"expected one of {sorted(chunks_by_id)}"
        )
    return citation.model_copy(
        update={"document_id": chunk.document_id, "source_uri": _source_uri(chunk)}
    )


def _build_prompt(state: AgentState) -> str:
    """Build the synthesis prompt from the original question and numbered context chunks.

    Args:
        state: Agent state providing ``original_query`` and ``retrieved_chunks``.

    Returns:
        Formatted prompt with labelled chunks the LLM can cite.
    """
    formatted_chunks = "\n\n".join(
        (
            f"[{idx}] chunk_id={chunk.id} document_id={chunk.document_id} "
            f"source_uri={_source_uri(chunk)}\n{chunk.text}"
        )
        for idx, chunk in enumerate(state.retrieved_chunks, start=1)
    )
    return f"Question:\n{state.original_query}\n\nContext:\n{formatted_chunks}\n\nAnswer:"
