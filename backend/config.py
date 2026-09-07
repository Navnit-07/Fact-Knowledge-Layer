import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("VF_DATA_DIR", str(BASE_DIR / "data")))
PDF_STORE_DIR = DATA_DIR / "pdfs"
VECTOR_DIR = DATA_DIR / "vectors"
SQLITE_PATH = DATA_DIR / "verifact.db"

PDF_STORE_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)


GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", os.getenv("OPENAI_API_KEY", "")))
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

EXTRACTION_MODEL    = os.getenv("VF_EXTRACTION_MODEL",    "gemini-1.5-flash")
RECONCILIATION_MODEL = os.getenv("VF_RECONCILIATION_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL     = os.getenv("VF_EMBEDDING_MODEL",     "text-embedding-004")


CHUNK_MAX_CHARS    = int(os.getenv("VF_CHUNK_MAX_CHARS",    "6000"))
CHUNK_OVERLAP_RATIO = float(os.getenv("VF_CHUNK_OVERLAP_RATIO", "0.12"))
CHUNK_OVERLAP_CHARS = int(os.getenv("VF_CHUNK_OVERLAP_CHARS",
                                    str(int(CHUNK_MAX_CHARS * CHUNK_OVERLAP_RATIO))))


MIN_CONFIDENCE = float(os.getenv("VF_MIN_CONFIDENCE", "0.30"))


CHUNK_CONCURRENCY    = int(os.getenv("VF_CHUNK_CONCURRENCY",    "5"))
DOCUMENT_CONCURRENCY = int(os.getenv("VF_DOCUMENT_CONCURRENCY", "3"))
RECONCILE_CONCURRENCY = int(os.getenv("VF_RECONCILE_CONCURRENCY", "5"))


SIMILARITY_THRESHOLD = float(os.getenv("VF_SIMILARITY_THRESHOLD", "0.62"))
CANDIDATES_PER_FACT  = int(os.getenv("VF_CANDIDATES_PER_FACT",    "6"))
MAX_CLUSTER_SIZE     = int(os.getenv("VF_MAX_CLUSTER_SIZE",        "5"))

VECTOR_COLLECTION = "verifact_facts"
