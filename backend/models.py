from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RawFactOutput(BaseModel):

    entity: str = Field(
        ...,
        description="Primary subject of the fact, e.g. 'Delhivery', 'India CPI', 'RBI Governor'.",
    )
    metric_or_claim: str = Field(
        ...,
        description="What is being stated about the entity, e.g. 'FY24 revenue', 'Director since', 'appointed CEO'.",
    )
    value: Optional[str] = Field(
        None,
        description="The numeric or categorical value as written, e.g. '7,225 crore', '5.1%', 'resigned'.",
    )
    unit: Optional[str] = Field(
        None,
        description="Unit or currency of the value, e.g. 'INR crore', '%', 'USD million'.",
    )
    time_period: Optional[str] = Field(
        None,
        description="Period the fact applies to, e.g. 'FY24', 'Q4 FY24', 'as of 31 March 2025'.",
    )
    context: str = Field(
        ...,
        description="One or two sentences that make this fact self-contained without re-reading the source.",
    )
    exact_quote: str = Field(
        ...,
        description="A verbatim substring copied exactly from the source text. Must be findable character-for-character in the chunk.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model confidence that this fact is correctly extracted and grounded in the quote.",
    )
    extra: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional domain-specific attributes that don't fit the fixed fields (e.g. scope, segment, currency basis).",
    )


class ChunkExtractionOutput(BaseModel):

    facts: List[RawFactOutput]
    extraction_issues: List[str] = Field(
        default_factory=list,
        description=(
            "Notes on content that looked fact-like but couldn't be reliably extracted "
            "(garbled tables, unresolvable references, ambiguous figures)."
        ),
    )


class Fact(BaseModel):
    fact_id:         str          = Field(default_factory=lambda: new_id("fact"))
    document_id:     str
    source_filename: str
    page_number:     int
    chunk_id:        str
    entity:          str
    metric_or_claim: str
    value:           Optional[str] = None
    unit:            Optional[str] = None
    time_period:     Optional[str] = None
    context:         str
    exact_quote:     str
    confidence:      float
    grounded:        bool = True
    extra:           Dict[str, str] = Field(default_factory=dict)

    def canonical_text(self) -> str:
        parts = [self.entity, self.metric_or_claim]
        if self.value:
            parts.append(f"{self.value} {self.unit or ''}".strip())
        if self.time_period:
            parts.append(f"({self.time_period})")
        parts.append(self.context)
        return " | ".join(p for p in parts if p)


class ExtractionFailure(BaseModel):
    failure_id:      str          = Field(default_factory=lambda: new_id("fail"))
    document_id:     str
    source_filename: str
    page_number:     int
    chunk_id:        str
    reason:          str
    raw_excerpt:     Optional[str] = None


class RelationshipType(str, Enum):
    CORROBORATED       = "CORROBORATED"
    CONTRADICTION      = "CONTRADICTION"
    RESOLVED_BY_CONTEXT = "RESOLVED_BY_CONTEXT"
    EXTRACTION_FAILURE = "EXTRACTION_FAILURE"
    UNRELATED          = "UNRELATED"


class ClusterJudgment(BaseModel):
    relationship_type: RelationshipType
    involved_fact_indices: List[int] = Field(
        ...,
        description="0-based indices into the fact list given in the prompt. Usually 2, occasionally more.",
    )
    explanation: str = Field(
        ...,
        description="Plain-language reasoning citing time period, unit, scope, or entity differences.",
    )
    resolution_context: Optional[str] = Field(
        None,
        description="For RESOLVED_BY_CONTEXT: the specific differentiator (e.g. 'standalone vs consolidated', 'FY23 vs FY24').",
    )


class ClusterAnalysis(BaseModel):
    relationships: List[ClusterJudgment]


class FactRelationship(BaseModel):
    relationship_id:   str               = Field(default_factory=lambda: new_id("rel"))
    relationship_type: RelationshipType
    fact_ids:          List[str]
    explanation:       str
    resolution_context: Optional[str]   = None
    cluster_key:       str


class DocumentRecord(BaseModel):
    document_id:  str = Field(default_factory=lambda: new_id("doc"))
    filename:     str
    page_count:   int
    fact_count:   int = 0
    status:       str = "processing"
    ingested_at:  str = Field(default_factory=utc_now)


class SystemStats(BaseModel):
    total_documents:    int
    total_facts:        int
    grounded_facts:     int
    ungrounded_facts:   int
    total_failures:     int
    total_relationships: int
    relationships_by_type: Dict[str, int]
