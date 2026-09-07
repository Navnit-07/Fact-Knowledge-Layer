from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import List, Tuple

import pdfplumber
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from . import config
from .models import ChunkExtractionOutput, ExtractionFailure, Fact, RawFactOutput, new_id

_client: AsyncOpenAI | None = None

CHUNK_CONCURRENCY = config.CHUNK_CONCURRENCY


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        kwargs: dict = {"api_key": config.GEMINI_API_KEY}
        if config.GEMINI_BASE_URL:
            kwargs["base_url"] = config.GEMINI_BASE_URL
        _client = AsyncOpenAI(**kwargs)
    return _client


def _clean_json_response(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    pages: List[PageText]


def read_pdf_pages(pdf_path: str) -> List[PageText]:
    pages: List[PageText] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(PageText(page_number=i, text=text))
    return pages


def build_chunks(pages: List[PageText]) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    current_pages: List[PageText] = []
    current_len = 0
    overlap_tail = ""

    def _flush() -> None:
        nonlocal current_pages, current_len, overlap_tail
        if not current_pages:
            return
        text = overlap_tail + "\n".join(p.text for p in current_pages)
        chunks.append(TextChunk(chunk_id=new_id("chunk"), text=text, pages=list(current_pages)))
        overlap_tail = text[-config.CHUNK_OVERLAP_CHARS :]
        current_pages.clear()
        current_len = 0

    for page in pages:
        if not page.text.strip():
            continue
        if current_len + len(page.text) > config.CHUNK_MAX_CHARS and current_pages:
            _flush()
        current_pages.append(page)
        current_len += len(page.text)

    _flush()
    return chunks


_EXTRACTION_SYSTEM_PROMPT = """\
You are a precise fact-extraction engine for a document intelligence system.

Your job: read the supplied text chunk and pull out every discrete, verifiable
claim — numeric figures, dates, named entities and their roles, performance
statements, appointments, events. Let the document's own content decide what
counts as a fact; do not impose a fixed taxonomy.

Hard rules:
1. `exact_quote` must be copied character-for-character from the provided text.
   You may trim words around the span but must not rephrase the quoted portion.
   A quote that cannot be found verbatim in the source is worse than no quote.
2. Prefer several tight, precise facts over one broad paraphrase.
3. Skip pure boilerplate (page numbers, legal disclaimers, headers) unless they
   contain a verifiable claim.
4. When something looks fact-like but is too garbled, table-mangled, or
   ambiguous to extract confidently, skip it and add a note to
   `extraction_issues` instead of guessing.
5. Use `extra` for attributes that matter for this fact but don't fit the fixed
   fields — e.g. {"scope": "consolidated", "segment": "express-parcel",
   "currency_basis": "INR crore"}.
6. Set `confidence` honestly: clean prose → high; inferred from a noisy table
   → lower.

Output format (JSON only, no prose):
{
  "facts": [
    {
      "entity": "string",
      "metric_or_claim": "string",
      "value": "string | null",
      "unit": "string | null",
      "time_period": "string | null",
      "context": "string",
      "exact_quote": "string",
      "confidence": 0.0–1.0,
      "extra": {}
    }
  ],
  "extraction_issues": ["string"]
}
"""


def _build_user_message(chunk_text: str, filename: str) -> str:
    return (
        f"Source document: {filename}\n\n"
        f'Text chunk:\n"""\n{chunk_text}\n"""\n\n'
        "Extract all facts from this chunk as JSON."
    )


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
async def _call_extraction_llm(chunk_text: str, filename: str) -> ChunkExtractionOutput:
    client = _get_client()
    completion = await client.chat.completions.create(
        model=config.EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_message(chunk_text, filename)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("LLM returned an empty response for extraction.")
    return ChunkExtractionOutput.model_validate_json(_clean_json_response(content))


def _normalise(s: str) -> str:
    s = re.sub(r"-\s*\n\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _is_grounded(quote: str, chunk_text: str) -> bool:
    return bool(quote.strip()) and _normalise(quote) in _normalise(chunk_text)


def _best_page_for_quote(quote: str, pages: List[PageText]) -> int:
    norm_q = _normalise(quote)
    if norm_q:
        for p in pages:
            if norm_q in _normalise(p.text):
                return p.page_number
    return pages[0].page_number if pages else -1


async def mine_chunk_for_facts(
    chunk: TextChunk,
    document_id: str,
    filename: str,
) -> Tuple[List[Fact], List[ExtractionFailure]]:
    facts: List[Fact] = []
    failures: List[ExtractionFailure] = []

    try:
        result = await _call_extraction_llm(chunk.text, filename)
    except Exception as exc:
        failures.append(
            ExtractionFailure(
                document_id=document_id,
                source_filename=filename,
                page_number=chunk.pages[0].page_number if chunk.pages else -1,
                chunk_id=chunk.chunk_id,
                reason=f"LLM extraction call failed: {exc}",
                raw_excerpt=chunk.text[:300],
            )
        )
        return facts, failures


    for note in result.extraction_issues:
        failures.append(
            ExtractionFailure(
                document_id=document_id,
                source_filename=filename,
                page_number=chunk.pages[0].page_number if chunk.pages else -1,
                chunk_id=chunk.chunk_id,
                reason=note,
            )
        )

    for raw in result.facts:

        if raw.confidence < config.MIN_CONFIDENCE:
            failures.append(
                ExtractionFailure(
                    document_id=document_id,
                    source_filename=filename,
                    page_number=chunk.pages[0].page_number if chunk.pages else -1,
                    chunk_id=chunk.chunk_id,
                    reason=(
                        f"Fact '{raw.entity} / {raw.metric_or_claim}' auto-rejected: "
                        f"confidence {raw.confidence:.2f} is below threshold {config.MIN_CONFIDENCE:.2f}."
                    ),
                    raw_excerpt=raw.exact_quote,
                )
            )
            continue

        grounded = _is_grounded(raw.exact_quote, chunk.text)
        page_num = _best_page_for_quote(raw.exact_quote, chunk.pages)

        fact = Fact(
            document_id=document_id,
            source_filename=filename,
            page_number=page_num,
            chunk_id=chunk.chunk_id,
            entity=raw.entity,
            metric_or_claim=raw.metric_or_claim,
            value=raw.value,
            unit=raw.unit,
            time_period=raw.time_period,
            context=raw.context,
            exact_quote=raw.exact_quote,
            confidence=raw.confidence,
            grounded=grounded,
            extra=raw.extra,
        )
        facts.append(fact)

        if not grounded:
            failures.append(
                ExtractionFailure(
                    document_id=document_id,
                    source_filename=filename,
                    page_number=page_num,
                    chunk_id=chunk.chunk_id,
                    reason=(
                        f"Fact '{fact.entity} / {fact.metric_or_claim}' kept but marked ungrounded: "
                        "exact_quote could not be found verbatim in the source chunk "
                        "(likely paraphrase or whitespace/OCR mismatch)."
                    ),
                    raw_excerpt=raw.exact_quote,
                )
            )

    return facts, failures


async def process_document(
    pdf_path: str,
    filename: str,
    document_id: str,
) -> Tuple[List[Fact], List[ExtractionFailure], int]:

    try:
        pages = await asyncio.to_thread(read_pdf_pages, pdf_path)
    except Exception as exc:
        return (
            [],
            [ExtractionFailure(
                document_id=document_id, source_filename=filename,
                page_number=-1, chunk_id="n/a",
                reason=f"PDF could not be opened or parsed: {exc}",
            )],
            0,
        )

    if not pages:
        return (
            [],
            [ExtractionFailure(
                document_id=document_id, source_filename=filename,
                page_number=-1, chunk_id="n/a",
                reason="PDF opened but contained no pages.",
            )],
            0,
        )

    chunks = build_chunks(pages)
    if not chunks:
        return (
            [],
            [ExtractionFailure(
                document_id=document_id, source_filename=filename,
                page_number=-1, chunk_id="n/a",
                reason="PDF has pages but no extractable text (scanned/image-only with no OCR layer).",
            )],
            len(pages),
        )

    semaphore = asyncio.Semaphore(CHUNK_CONCURRENCY)

    async def _bounded(chunk: TextChunk):
        async with semaphore:
            return await mine_chunk_for_facts(chunk, document_id, filename)

    results = await asyncio.gather(*(_bounded(c) for c in chunks))

    all_facts: List[Fact] = []
    all_failures: List[ExtractionFailure] = []
    for f_list, fail_list in results:
        all_facts.extend(f_list)
        all_failures.extend(fail_list)

    return all_facts, all_failures, len(pages)