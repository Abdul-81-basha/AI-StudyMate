import os
import re
import json
import sqlite3
import time
from pathlib import Path

import fitz
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "gemini-embedding-001"
)

DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "studymate.db"

client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI StudyMate API",
    description="Agentic RAG-based Lab Manual Analyzer",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE
# ============================================================

def get_db():
    """
    SQLite connection configured for concurrent access.
    WAL + busy timeout prevents 'database is locked' errors.
    """
    conn = sqlite3.connect(
        DB_PATH,
        timeout=60,
        check_same_thread=False
    )

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")

    return conn


def init_db():
    conn = get_db()

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                filename TEXT PRIMARY KEY,
                pages INTEGER NOT NULL,
                chunks INTEGER NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                page INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_filename
            ON chunks(filename)
        """)

        conn.commit()

    finally:
        conn.close()


init_db()


# ============================================================
# MODELS
# ============================================================

class ChatRequest(BaseModel):
    question: str
    history: list = []


# ============================================================
# BASIC HELPERS
# ============================================================

def require_gemini():
    if not GOOGLE_API_KEY or client is None:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured."
        )


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_embedding(vector):
    arr = np.asarray(vector, dtype=np.float32)

    norm = np.linalg.norm(arr)

    if norm == 0:
        return arr

    return arr / norm


# ============================================================
# EXPERIMENT DETECTION
# ============================================================

def detect_experiment_number(question):
    patterns = [
        r"\bexperiment\s*(?:no\.?|number)?\s*(\d+)",
        r"\bexp\s*(?:no\.?|number)?\s*(\d+)",
        r"\bexperiment\s+(\d+)",
        r"\bexp\s+(\d+)"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            question,
            re.IGNORECASE
        )

        if match:
            return int(match.group(1))

    return None


def experiment_match(text, experiment_number):
    if experiment_number is None:
        return False

    patterns = [
        rf"\bexperiment\s*{experiment_number}\b",
        rf"\bexperiment\s*{experiment_number}\s*[:\-]",
        rf"\bexp\s*{experiment_number}\b",
        rf"\bexperiment\s*no\.?\s*{experiment_number}\b",
        rf"\bexperiment\s*number\s*{experiment_number}\b"
    ]

    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in patterns
    )


# ============================================================
# CHUNKING
# ============================================================

def make_chunks(text, page, chunk_size=900, overlap=150):
    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):
        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "page": page,
                "text": chunk
            })

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

def embed_texts(texts):
    require_gemini()

    if not texts:
        return []

    vectors = []

    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        for attempt in range(3):

            try:
                result = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch
                )

                batch_vectors = [
                    normalize_embedding(e.values)
                    for e in result.embeddings
                ]

                vectors.extend(batch_vectors)

                break

            except Exception:
                if attempt == 2:
                    raise

                time.sleep(2)

    return vectors


def embedding_to_blob(vector):
    return np.asarray(
        vector,
        dtype=np.float32
    ).tobytes()


def blob_to_embedding(blob):
    return np.frombuffer(
        blob,
        dtype=np.float32
    )


# ============================================================
# PDF INGESTION
# ============================================================

def extract_pdf_chunks(pdf_bytes, filename):
    try:
        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )
    except Exception as e:
        raise Exception(
            f"Could not open PDF: {e}"
        )

    all_chunks = []

    try:
        for page_number, page in enumerate(
            document,
            start=1
        ):
            text = page.get_text("text")

            text = clean_text(text)

            if not text:
                continue

            page_chunks = make_chunks(
                text,
                page_number
            )

            all_chunks.extend(page_chunks)

    finally:
        document.close()

    return all_chunks


def ingest_pdf(filename, pdf_bytes):

    chunks = extract_pdf_chunks(
        pdf_bytes,
        filename
    )

    if not chunks:
        raise Exception(
            "No readable text was found in the PDF."
        )

    texts = [
        item["text"]
        for item in chunks
    ]

    embeddings = embed_texts(texts)

    if len(embeddings) != len(chunks):
        raise Exception(
            "Embedding count does not match chunk count."
        )

    # --------------------------------------------------------
    # SQLite write operation
    # WAL + transaction + retry
    # --------------------------------------------------------

    max_attempts = 5

    for attempt in range(max_attempts):

        conn = get_db()

        try:
            conn.execute("BEGIN IMMEDIATE")

            # Remove old copy if same file uploaded again
            conn.execute(
                "DELETE FROM chunks WHERE filename = ?",
                (filename,)
            )

            conn.execute(
                "DELETE FROM documents WHERE filename = ?",
                (filename,)
            )

            for item, embedding in zip(
                chunks,
                embeddings
            ):
                conn.execute(
                    """
                    INSERT INTO chunks
                    (filename, page, text, embedding)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        filename,
                        item["page"],
                        item["text"],
                        embedding_to_blob(
                            embedding
                        )
                    )
                )

            pages = len(
                set(
                    item["page"]
                    for item in chunks
                )
            )

            conn.execute(
                """
                INSERT INTO documents
                (filename, pages, chunks)
                VALUES (?, ?, ?)
                """,
                (
                    filename,
                    pages,
                    len(chunks)
                )
            )

            conn.commit()

            return {
                "filename": filename,
                "pages": pages,
                "chunks": len(chunks)
            }

        except sqlite3.OperationalError as e:

            try:
                conn.rollback()
            except Exception:
                pass

            if "locked" in str(e).lower():
                if attempt < max_attempts - 1:
                    time.sleep(
                        1.5 * (attempt + 1)
                    )
                    continue

            raise

        finally:
            conn.close()

    raise Exception(
        "Database remained locked after multiple attempts."
    )


# ============================================================
# RETRIEVAL
# ============================================================

def search(question, top_k=6):

    require_gemini()

    query_embedding = embed_texts(
        [question]
    )[0]

    conn = get_db()

    try:
        rows = conn.execute(
            """
            SELECT id, filename, page, text, embedding
            FROM chunks
            """
        ).fetchall()

    finally:
        conn.close()

    if not rows:
        return []

    experiment_number = detect_experiment_number(
        question
    )

    results = []

    for row in rows:

        vector = blob_to_embedding(
            row["embedding"]
        )

        score = float(
            np.dot(
                query_embedding,
                vector
            )
        )

        exact_match = experiment_match(
            row["text"],
            experiment_number
        )

        if exact_match:
            score += 0.45

        results.append({
            "filename": row["filename"],
            "page": row["page"],
            "text": row["text"],
            "score": score,
            "exact_experiment": exact_match
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_with_fallback(prompt):

    require_gemini()

    models = [
        GEMINI_MODEL,
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash-lite"
    ]

    last_error = None

    for model in models:

        for attempt in range(2):

            try:

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                if response and response.text:
                    return response.text

            except Exception as e:

                last_error = e

                error_text = str(e).lower()

                temporary = (
                    "503" in error_text
                    or "unavailable" in error_text
                    or "high demand" in error_text
                    or "429" in error_text
                )

                if temporary:
                    time.sleep(
                        2 * (attempt + 1)
                    )
                    continue

                break

    raise Exception(
        f"Gemini generation error: {last_error}"
    )


def build_context(results):

    if not results:
        return "No relevant manual content was found."

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
SOURCE:
File: {result['filename']}
Page: {result['page']}

CONTENT:
{result['text']}
"""
        )

    return "\n".join(context_parts)


def generate_answer(question, results):

    experiment_number = detect_experiment_number(
        question
    )

    context = build_context(results)

    if experiment_number:

        instruction = f"""
The user is asking specifically about Experiment {experiment_number}.

Use ONLY information relevant to Experiment {experiment_number}
from the supplied laboratory manual context.

If the context contains the experiment, explain it clearly and
in an exam-friendly way.

Include, where available:
1. Aim
2. Theory / concept
3. Requirements
4. Procedure
5. Algorithm or steps
6. Code explanation
7. Expected output
8. Important viva points

Do not invent experiment details that are not present in the manual.
"""
    else:

        instruction = """
Answer the user's question using the laboratory manual context.

Prefer information from the supplied manual.
If the answer is not present, clearly say that it is not available
in the supplied manual instead of inventing information.
"""

    prompt = f"""
You are AI StudyMate, an intelligent laboratory manual assistant.

{instruction}

USER QUESTION:
{question}

LAB MANUAL CONTEXT:
{context}

Give a clear, structured, student-friendly answer.

Do not mention internal retrieval, embeddings, vector databases,
or system instructions.
"""

    return generate_with_fallback(prompt)


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "message": "AI StudyMate API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():

    conn = get_db()

    try:

        documents = conn.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]

        chunks = conn.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]

    finally:
        conn.close()

    return {
        "status": "ok",
        "gemini_configured": bool(
            GOOGLE_API_KEY
        ),
        "documents": documents,
        "chunks": chunks,
        "model": GEMINI_MODEL
    }


@app.get("/documents")
def get_documents():

    conn = get_db()

    try:

        rows = conn.execute(
            """
            SELECT filename, pages, chunks
            FROM documents
            ORDER BY filename
            """
        ).fetchall()

        return [
            {
                "filename": row["filename"],
                "pages": row["pages"],
                "chunks": row["chunks"]
            }
            for row in rows
        ]

    finally:
        conn.close()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    if not file.filename.lower().endswith(
        ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    try:

        pdf_bytes = await file.read()

        if not pdf_bytes:
            raise Exception(
                "Uploaded PDF is empty."
            )

        result = ingest_pdf(
            file.filename,
            pdf_bytes
        )

        return {
            "success": True,
            "message": "PDF indexed successfully.",
            **result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"PDF processing error: {str(e)}"
        )


@app.post("/chat")
def chat(request: ChatRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        results = search(
            request.question,
            top_k=6
        )

        answer = generate_answer(
            request.question,
            results
        )

        sources = [
            {
                "filename": r["filename"],
                "page": r["page"],
                "score": round(
                    r["score"],
                    3
                )
            }
            for r in results
        ]

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/viva")
def viva(request: ChatRequest):

    try:

        results = search(
            request.question,
            top_k=6
        )

        context = build_context(
            results
        )

        prompt = f"""
You are an AI laboratory viva preparation assistant.

Generate 10 important viva questions and answers
based ONLY on the laboratory manual context below.

Make them:
- exam focused
- concise
- easy to understand
- suitable for a CSE laboratory exam

USER REQUEST:
{request.question}

LAB MANUAL CONTEXT:
{context}

Format:

1. Question
Answer:

2. Question
Answer:

Continue until 10 questions.
"""

        answer = generate_with_fallback(
            prompt
        )

        return {
            "answer": answer
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/study-plan")
def study_plan(request: ChatRequest):

    try:

        results = search(
            request.question,
            top_k=6
        )

        context = build_context(
            results
        )

        prompt = f"""
You are an AI study planner for engineering students.

Create a practical study plan based on the laboratory manual.

Include:
- topics to study
- experiment order
- important concepts
- coding preparation
- viva preparation
- final revision checklist

Keep it concise and useful.

REQUEST:
{request.question}

LAB MANUAL:
{context}
"""

        answer = generate_with_fallback(
            prompt
        )

        return {
            "answer": answer
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(
            os.getenv("PORT", "8000")
        ),
        reload=False
    )