# VeriFact: Autonomous Multi-Document Fact Extraction & Cross-Verification

VeriFact is an automated knowledge extraction, grounding, and reconciliation pipeline that ingests unstructured PDF reports, extracts granular factual claims with strict verbatim grounding, and cross-references them across documents to uncover agreements, direct contradictions, and contextually explainable variances.

---

## Setup and Run Instructions

### Prerequisites
- Python 3.11+
- Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))

### 1. Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/Navnak/fact-knowledge.git
cd fact-knowledge

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and supply your Gemini API key:

```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
VF_API_BASE=http://localhost:8000
VF_EXTRACTION_MODEL=gemini-1.5-flash
VF_RECONCILIATION_MODEL=gemini-1.5-flash
VF_EMBEDDING_MODEL=text-embedding-004
```

### 3. Launch the Backend Server

Start the FastAPI application with Uvicorn:

```bash
uvicorn backend.main:app --reload --port 8000
```
- API will be accessible at: `http://localhost:8000`
- Interactive OpenAPI / Swagger Documentation: `http://localhost:8000/docs`

### 4. Launch the Frontend UI

In a new terminal (with the virtual environment activated), start the Streamlit dashboard:

```bash
streamlit run frontend/app.py
```
- Access the web console at: `http://localhost:8501`
- Navigate to the **Ingest** tab, upload one or more PDFs, and execute the pipeline.

---

## Video Demo

[![VeriFact Demo Video](https://img.shields.io/badge/Demo_Video-Watch_Walkthrough-blue?style=for-the-badge&logo=youtube)](https://drive.google.com/file/d/1wBNngnv-ZR91iNmlF7qzKZaHWQkrNjqs/view?usp=drive_link)

> **Demo Video Link:** [https://drive.google.com/file/d/1wBNngnv-ZR91iNmlF7qzKZaHWQkrNjqs/view?usp=drive_link](https://drive.google.com/file/d/1wBNngnv-ZR91iNmlF7qzKZaHWQkrNjqs/view?usp=drive_link)  
> *A concise walkthrough (< 3 minutes) demonstrating document ingestion, fact extraction, and live detection across the four core relationships:*
> 1. **CORROBORATED**: Mutual agreement across multiple sources on identical metrics.
> 2. **CONTRADICTION**: Direct metric discrepancies under identical scopes and periods.
> 3. **RESOLVED_BY_CONTEXT**: Resolution of apparent divergence due to standalone vs. consolidated reporting or reporting boundaries.
> 4. **EXTRACTION_FAILURE**: Identification and audit isolation of ungrounded or ambiguous claims.

---

## Approach

### 1. Architecture Overview

VeriFact follows an asynchronous, pipeline-driven architecture:

```
[ PDF Documents ]
       │
       ▼
[ Extractor (pdfplumber) ] ────> Sliding-Window Text Chunks (Overlap: 200 chars)
       │
       ▼
[ Gemini 1.5 Flash ] ──────────> Structured Facts (Entity, Claim, Value, Unit, Period, Exact Quote)
       │
       ├───> [ Grounding Validator ] ──> Verbatim Substring Match Check
       │
       ├───> [ SQLite Store ] ─────────> Document & Fact Persistence
       │
       ▼
[ Gemini Embeddings ] ─────────> [ ChromaDB Vector Index ] (text-embedding-004)
       │
       ▼
[ Candidate Pair Generator ] ──> Cosine Similarity Filtering (Threshold: 0.82)
       │
       ▼
[ Graph Clustering ] ──────────> Disjoint Set / Union-Find Clustering
       │
       ▼
[ Gemini Adjudicator ] ────────> Cross-Document Relationship Judgment
       │
       ▼
[ Streamlit Cockpit ] ─────────> Visual Findings, Evidence Diffs & Distribution Analytics
```

### 2. Core Decisions & Pipeline Stages

1. **Sliding-Window PDF Parsing & Extraction**:
   - Uses `pdfplumber` for text extraction. Text is segmented into chunks with an overlap buffer (`CHUNK_OVERLAP_CHARS = 200`) to guarantee metrics spanning across page boundaries are not lost.
   - Fact extraction leverages `gemini-1.5-flash` with a JSON schema prompt to pull granular facts: `entity`, `metric_or_claim`, `value`, `unit`, `time_period`, `context`, `exact_quote`, and a model-assessed `confidence` score.

2. **Strict Verbatim Source Grounding**:
   - To eliminate LLM hallucinations, every extracted fact undergoes a deterministic post-extraction verification step: the extracted `exact_quote` must exist verbatim in the source chunk text.
   - If missing, the fact is marked `grounded = False` and an `ExtractionFailure` record is created for transparency.

3. **Two-Tier Reconciliation Engine (ChromaDB + Union-Find)**:
   - Comparing every fact pair via LLM is $O(N^2)$ and computationally prohibitive.
   - Instead, facts are embedded using Google's `text-embedding-004` and indexed in ChromaDB.
   - VeriFact queries nearest neighbors with cosine similarity thresholding ($k=8$, similarity $\ge 0.82$).
   - A **Disjoint Set (Union-Find)** data structure aggregates overlapping pairs into coherent multi-fact clusters.

4. **Multi-Class Fact Adjudication**:
   - Multi-fact clusters are passed to Gemini 1.5 Flash in a single prompt. The model evaluates whether the cluster exhibits:
     - `CORROBORATED`: Multiple sources validate the exact same figure or claim.
     - `CONTRADICTION`: Direct factual contradiction without contextual justification.
     - `RESOLVED_BY_CONTEXT`: Seeming divergence explained by different time periods, accounting standards (GAAP vs. IFRS), or entity scopes (consolidated vs. standalone).
     - `EXTRACTION_FAILURE`: Corrupted quotes or ungrounded claims.

### 3. Trade-offs & Decisions

| Decision | Alternative Considered | Why Selected |
|---|---|---|
| **Two-Tier Clustering (Vector + Graph)** | All-pairs $O(N^2)$ LLM calls | Reduces API overhead by 95%+ while retaining high semantic recall for candidate pairs. |
| **Verbatim Substring Grounding** | LLM-based self-verification | Deterministic substring check runs in sub-millisecond time and eliminates confirmation bias from the LLM. |
| **Pydantic Validation + JSON Sanitizing** | Freeform markdown parsing | Robust schema validation with automatic code-fence stripping ensures high reliability in production pipelines. |
| **Streamlit + FastAPI Hybrid** | Monolithic frontend/backend | Allows the backend to function as a standalone microservice while offering an interactive inspection UI. |

### 4. AI Models & Tools Used
- **Google Gemini 1.5 Flash**: Primary inference engine for high-speed, cost-effective fact extraction and multi-fact reconciliation.
- **text-embedding-004**: Generates 768-dimensional dense vector representations for semantic search.
- **ChromaDB**: In-memory vector database with persistence.
- **Pydantic v2**: Strict schema definition and type safety.
- **FastAPI & Uvicorn**: High-concurrency async API server.
- **Streamlit & Plotly**: Dark slate analytics cockpit and distribution visualizations.

---

## Limitations and Next Steps

### Current Limitations
1. **Scanned / Rasterized PDFs**: Currently relies on digital text layers via `pdfplumber`. PDFs containing scanned images or bitmaps require an upstream OCR engine (e.g., Tesseract or Google Cloud Vision).
2. **Complex Multi-Column / Nested Tables**: While sliding-window chunking captures standard tabular flows, complex financial tables with merged multi-level headers can lose row/column associations.
3. **Cross-Language Discrepancy**: Embeddings and reconciliation prompts are tuned for English documents; cross-lingual extraction is not yet benchmarked.

### Roadmap & Next Steps
- [ ] **Native Document Layout OCR**: Integrate layout-aware vision models to extract complex nested tables and spatial relationships.
- [ ] **Temporal Knowledge Graph**: Store verified facts in a directed property graph (e.g., Neo4j) to track metric drift across fiscal quarters.
- [ ] **Multi-Hop Reasoning**: Support transitive reconciliation (e.g., Document A relates to B, and B relates to C $\rightarrow$ verify A with C).
- [ ] **Real-Time Webhook Notifications**: Alert external audit systems whenever a critical `CONTRADICTION` is surfaced.

---

## Additional Notes

- **Deterministic Fallbacks**: Every asynchronous LLM operation is wrapped in exponential backoff retry mechanisms (`tenacity`) to gracefully handle transient rate limits or network spikes.
- **Data Persistence**: All ingested documents, extracted facts, and discovered relationships persist locally inside `./data/` (`facts.db` and ChromaDB vector files), ensuring state survives restarts.
- **Audit Transparency**: Discrepancies are accompanied by verbatim quotes, page numbers, and model rationales, providing human-in-the-loop verification.

---

## License

This project is licensed under the MIT License.
