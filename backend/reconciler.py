from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Set, Tuple

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from . import config
from .models import ClusterAnalysis, ClusterJudgment, Fact, FactRelationship, RelationshipType

_chroma: Any | None = None
_openai: AsyncOpenAI | None = None


def _clean_json_response(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _get_client() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        kwargs: dict = {"api_key": config.GEMINI_API_KEY}
        if config.GEMINI_BASE_URL:
            kwargs["base_url"] = config.GEMINI_BASE_URL
        _openai = AsyncOpenAI(**kwargs)
    return _openai


def _get_collection():
    global _chroma
    if _chroma is None:
        import chromadb
        _chroma = chromadb.PersistentClient(path=str(config.VECTOR_DIR))
    return _chroma.get_or_create_collection(
        name=config.VECTOR_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
async def _embed(texts: List[str]) -> List[List[float]]:
    resp = await _get_client().embeddings.create(
        model=config.EMBEDDING_MODEL, input=texts
    )
    return [d.embedding for d in resp.data]


async def embed_and_store_facts(facts: List[Fact]) -> None:
    if not facts:
        return
    collection = _get_collection()
    vectors = await _embed([f.canonical_text() for f in facts])
    await asyncio.to_thread(
        collection.upsert,
        ids=[f.fact_id for f in facts],
        embeddings=vectors,
        metadatas=[
            {
                "document_id":     f.document_id,
                "entity":          f.entity,
                "source_filename": f.source_filename,
            }
            for f in facts
        ],
        documents=[f.canonical_text() for f in facts],
    )


async def _neighbours_for(fact: Fact) -> List[Tuple[str, float]]:
    collection = _get_collection()
    vectors = await _embed([fact.canonical_text()])
    result = await asyncio.to_thread(
        collection.query,
        query_embeddings=vectors,
        n_results=config.CANDIDATES_PER_FACT + 1,
    )
    neighbours = []
    for nid, dist in zip(result["ids"][0], result["distances"][0]):
        if nid == fact.fact_id:
            continue
        similarity = 1.0 - dist
        if similarity >= config.SIMILARITY_THRESHOLD:
            neighbours.append((nid, similarity))
    return neighbours


async def cluster_related_facts(
    new_facts: List[Fact],
    all_facts_by_id: Dict[str, Fact],
) -> List[List[str]]:
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    neighbour_lists = await asyncio.gather(*(_neighbours_for(f) for f in new_facts))

    edges_seen: Set[Tuple[str, str]] = set()
    for fact, neighbours in zip(new_facts, neighbour_lists):
        parent.setdefault(fact.fact_id, fact.fact_id)
        for nid, _ in neighbours:
            if nid not in all_facts_by_id and nid not in {f.fact_id for f in new_facts}:
                continue
            edge = tuple(sorted((fact.fact_id, nid)))
            if edge in edges_seen:
                continue
            edges_seen.add(edge)
            union(fact.fact_id, nid)

    groups: Dict[str, List[str]] = {}
    involved = {n for edge in edges_seen for n in edge}
    for fid in involved:
        groups.setdefault(find(fid), []).append(fid)

    clusters = [sorted(set(members)) for members in groups.values() if len(members) >= 2]
    return [c[: config.MAX_CLUSTER_SIZE] for c in clusters]


_RECONCILIATION_SYSTEM_PROMPT = """\
You are a fact-reconciliation analyst for a document intelligence system.

You receive a small cluster of facts that vector similarity flagged as
potentially related. Your job is to judge the actual relationship between
them — whether they agree, disagree, or only appear to disagree.

Before labelling anything a CONTRADICTION, work through this checklist in order:
  1. Time period — do the facts cover different fiscal years, quarters, or dates?
  2. Unit / currency basis — crore vs million, percentage vs absolute, nominal vs real?
  3. Scope — standalone vs consolidated, one segment vs the whole entity, domestic vs global?
  4. Entity granularity — same name but different entities (subsidiary vs parent, individual vs org)?

If ANY of these factors plausibly explains the discrepancy, classify as
RESOLVED_BY_CONTEXT and name the specific differentiator in `resolution_context`
(e.g. "FY23 vs FY24", "standalone vs consolidated", "INR crore vs USD million").
Only reach for CONTRADICTION when the facts share the same time period, unit,
scope, and entity and still state incompatible things.

Classification options:
- CORROBORATED: facts assert the same underlying truth, possibly with different phrasing or rounding.
- CONTRADICTION: irreconcilable incompatibility on the same entity/metric/scope/period.
- RESOLVED_BY_CONTEXT: apparent conflict explained by one of the four factors above.
  You MUST populate `resolution_context`.
- EXTRACTION_FAILURE: the facts are too malformed, vague, or mismatched to judge responsibly.
- UNRELATED: embedding similarity was a false positive; these facts have nothing to do with each other.

In `explanation`, always state which of the four factors you checked and the
outcome for each — do not jump straight to a verdict.

Reference facts by their index only. If a cluster of >2 facts contains multiple
independent relationships (e.g. 0+1 corroborate, 2 is unrelated), output
multiple entries.

Output format (JSON only):
{
  "relationships": [
    {
      "fact_indices": [0, 1],
      "relationship_type": "CORROBORATED | CONTRADICTION | RESOLVED_BY_CONTEXT | EXTRACTION_FAILURE | UNRELATED",
      "explanation": "string",
      "resolution_context": "string or null"
    }
  ]
}
"""


def _format_cluster(facts: List[Fact]) -> str:
    blocks = []
    for i, f in enumerate(facts):
        blocks.append(
            f"[{i}] Source: {f.source_filename}  (page {f.page_number})\n"
            f"    Entity:       {f.entity}\n"
            f"    Metric/claim: {f.metric_or_claim}\n"
            f"    Value:        {f.value or 'n/a'} {f.unit or ''}\n"
            f"    Time period:  {f.time_period or 'n/a'}\n"
            f"    Context:      {f.context}\n"
            f"    Quote:        \"{f.exact_quote}\"\n"
            f"    Extra:        {f.extra or '{}'}"
        )
    return "\n\n".join(blocks)


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
async def _call_reconciliation_llm(facts: List[Fact]) -> ClusterAnalysis:
    client = _get_client()
    user_msg = (
        f"Facts to analyse:\n\n{_format_cluster(facts)}\n\n"
        "Judge the relationship(s) among these facts. Output valid JSON only matching the schema."
    )
    completion = await client.chat.completions.create(
        model=config.RECONCILIATION_MODEL,
        messages=[
            {"role": "system", "content": _RECONCILIATION_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("Reconciliation LLM returned no structured output.")
    return ClusterAnalysis.model_validate_json(_clean_json_response(content))


async def _judge_cluster(
    cluster_ids: List[str],
    facts_by_id: Dict[str, Fact],
) -> List[FactRelationship]:
    facts = [facts_by_id[fid] for fid in cluster_ids if fid in facts_by_id]
    if len(facts) < 2:
        return []

    analysis = await _call_reconciliation_llm(facts)
    results: List[FactRelationship] = []

    for judgment in analysis.relationships:
        if judgment.relationship_type == RelationshipType.UNRELATED:
            continue

        involved_ids = [
            facts[i].fact_id
            for i in judgment.involved_fact_indices
            if 0 <= i < len(facts)
        ]
        if len(involved_ids) < 2:
            continue

        cluster_key = "|".join(sorted(involved_ids))
        results.append(
            FactRelationship(
                relationship_type=judgment.relationship_type,
                fact_ids=involved_ids,
                explanation=judgment.explanation,
                resolution_context=judgment.resolution_context,
                cluster_key=cluster_key,
            )
        )
    return results


async def run_reconciliation(
    new_facts: List[Fact],
    all_facts: List[Fact],
) -> List[FactRelationship]:
    from . import db

    if not new_facts:
        return []

    await embed_and_store_facts(new_facts)

    all_facts_by_id: Dict[str, Fact] = {f.fact_id: f for f in all_facts}
    for f in new_facts:
        all_facts_by_id[f.fact_id] = f

    clusters = await cluster_related_facts(new_facts, all_facts_by_id)

    semaphore = asyncio.Semaphore(config.RECONCILE_CONCURRENCY)

    async def _bounded(cluster: List[str]) -> List[FactRelationship]:
        async with semaphore:
            return await _judge_cluster(cluster, all_facts_by_id)

    cluster_results = await asyncio.gather(*(_bounded(c) for c in clusters))

    new_relationships: List[FactRelationship] = []
    for rels in cluster_results:
        for rel in rels:
            if db.relationship_exists(rel.cluster_key):
                continue
            db.insert_relationship(rel)
            new_relationships.append(rel)

    return new_relationships
