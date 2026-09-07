from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config, db
from .extractor import process_document
from .models import (
    DocumentRecord,
    ExtractionFailure,
    Fact,
    FactRelationship,
    RelationshipType,
    SystemStats,
    new_id,
)
from .reconciler import run_reconciliation

logger = logging.getLogger("verifact")

app = FastAPI(
    title="VeriFact API",
    description=(
        "Upload PDFs, extract grounded facts, and surface cross-document "
        "corroborations, contradictions, and context-resolved conflicts."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    db.init_db()
    logger.info("Server started. SQLite: %s", config.SQLITE_PATH)


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "service": "verifact"}


@app.get("/stats", response_model=SystemStats, tags=["system"])
def get_stats():
    return db.get_stats()


async def _save_upload(upload: UploadFile, document_id: str):
    content = await upload.read()
    dest = config.PDF_STORE_DIR / f"{document_id}_{upload.filename}"
    await asyncio.to_thread(dest.write_bytes, content)
    return dest


async def _extract_one(
    upload: UploadFile,
    document_id: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    if not upload.filename.lower().endswith(".pdf"):
        return {
            "document_id": document_id,
            "filename": upload.filename,
            "ok": False,
            "error": "Only PDF files are accepted.",
            "facts": [],
            "failures": [
                ExtractionFailure(
                    document_id=document_id,
                    source_filename=upload.filename,
                    page_number=-1,
                    chunk_id="n/a",
                    reason="Rejected: not a .pdf file.",
                )
            ],
            "page_count": 0,
        }

    async with semaphore:
        try:
            dest_path = await _save_upload(upload, document_id)
            facts, failures, page_count = await process_document(
                str(dest_path), upload.filename, document_id
            )
        except Exception as exc:
            return {
                "document_id": document_id,
                "filename": upload.filename,
                "ok": False,
                "error": str(exc),
                "facts": [],
                "failures": [
                    ExtractionFailure(
                        document_id=document_id,
                        source_filename=upload.filename,
                        page_number=-1,
                        chunk_id="n/a",
                        reason=f"Unexpected error: {exc}",
                    )
                ],
                "page_count": 0,
            }

    return {
        "document_id": document_id,
        "filename": upload.filename,
        "ok": True,
        "error": None,
        "facts": facts,
        "failures": failures,
        "page_count": page_count,
    }


@app.post("/documents/upload", tags=["documents"])
async def upload_documents(files: List[UploadFile] = File(...)):
    if not config.GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set on the server. Add it to your .env file.",
        )

    document_ids = [new_id("doc") for _ in files]
    semaphore = asyncio.Semaphore(config.DOCUMENT_CONCURRENCY)
    extraction_results = await asyncio.gather(
        *(_extract_one(u, did, semaphore) for u, did in zip(files, document_ids))
    )

    summaries = []
    for result in extraction_results:
        document_id  = result["document_id"]
        filename     = result["filename"]
        facts: List[Fact] = result["facts"]
        failures: List[ExtractionFailure] = result["failures"]
        page_count   = result["page_count"]

        if not result["ok"]:
            db.upsert_document(
                DocumentRecord(document_id=document_id, filename=filename,
                               page_count=0, status="failed")
            )
            db.insert_failures(failures)
            summaries.append({
                "document_id": document_id, "filename": filename,
                "status": "failed", "error": result["error"],
                "page_count": 0, "facts_extracted": 0,
                "extraction_failures": len(failures), "new_relationships": 0,
            })
            continue

        db.insert_facts(facts)
        db.insert_failures(failures)

        status = "indexed" if facts else "failed"
        db.upsert_document(
            DocumentRecord(
                document_id=document_id, filename=filename,
                page_count=page_count, fact_count=len(facts), status=status,
            )
        )

        all_facts = db.list_facts()
        new_rels = await run_reconciliation(facts, all_facts) if facts else []

        summaries.append({
            "document_id": document_id, "filename": filename,
            "status": status,
            "error": None if facts else "No facts extracted (possibly a scanned/image PDF).",
            "page_count": page_count,
            "facts_extracted": len(facts),
            "extraction_failures": len(failures),
            "new_relationships": len(new_rels),
        })

    return {"processed": summaries}


@app.get("/documents", tags=["documents"])
def get_documents():
    return db.list_documents()


@app.delete("/documents/{document_id}", tags=["documents"])
def delete_document(document_id: str):
    docs = db.list_documents()
    if not any(d.document_id == document_id for d in docs):
        raise HTTPException(status_code=404, detail="Document not found.")
    removed_facts = db.delete_document(document_id)
    return {
        "document_id": document_id,
        "deleted": True,
        "facts_removed": removed_facts,
        "note": "Re-run /reconcile/rerun to refresh relationship scores if needed.",
    }


@app.get("/facts", response_model=List[Fact], tags=["facts"])
def get_facts(document_id: Optional[str] = None, entity: Optional[str] = None):
    return db.list_facts(document_id=document_id, entity=entity)


@app.get("/facts/{fact_id}", response_model=Fact, tags=["facts"])
def get_fact(fact_id: str):
    matches = db.get_facts_by_ids([fact_id])
    if not matches:
        raise HTTPException(status_code=404, detail="Fact not found.")
    return matches[0]


@app.get("/failures", tags=["failures"])
def get_failures(document_id: Optional[str] = None):
    return db.list_failures(document_id=document_id)


@app.get("/relationships", tags=["relationships"])
def get_relationships(relationship_type: Optional[str] = None):
    rels = db.list_relationships(relationship_type=relationship_type)

    return [
        {**rel.model_dump(), "facts": [f.model_dump() for f in db.get_facts_by_ids(rel.fact_ids)]}
        for rel in rels
    ]


@app.post("/reconcile/rerun", tags=["relationships"])
async def rerun_reconciliation():
    all_facts = db.list_facts()
    new_rels = await run_reconciliation(all_facts, all_facts)
    return {
        "facts_considered": len(all_facts),
        "new_relationships": len(new_rels),
    }
