import os
import sqlite3
import json
import time
import re
from pathlib import Path

import fitz
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

EMBED_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "gemini-embedding-001"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "studymate.db"

client = genai.Client(api_key=API_KEY) if API_KEY else None


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI StudyMate API",
    description="Agentic RAG based laboratory manual study assistant"
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
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            page INTEGER,
            text TEXT,
            embedding TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            filename TEXT PRIMARY KEY,
            pages INTEGER,
            chunks INTEGER
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ============================================================
# REQUEST MODEL
# ============================================================

class Req(BaseModel):
    question: str
    history: list = []


# ============================================================
# TEXT CHUNKING
# ============================================================

def make_chunks(text, size=900, overlap=150):

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = start + size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# GEMINI EMBEDDING
# ============================================================

def create_embedding(text):

    if not client:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is not configured."
        )

    try:

        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text
        )

        return result.embeddings[0].values

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Gemini embedding error: {str(e)}"
        )


# ============================================================
# PDF INGESTION
# ============================================================

def ingest_pdf(filename, pdf_bytes):

    if not client:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is not configured."
        )

    try:

        doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        all_chunks = []

        for page_number, page in enumerate(doc, start=1):

            text = page.get_text()

            chunks = make_chunks(text)

            for chunk in chunks:

                all_chunks.append({
                    "filename": filename,
                    "page": page_number,
                    "text": chunk
                })

        doc.close()

        if not all_chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable text was found in the PDF."
            )

        conn = get_db()

        conn.execute(
            "DELETE FROM chunks WHERE filename = ?",
            (filename,)
        )

        conn.execute(
            "DELETE FROM documents WHERE filename = ?",
            (filename,)
        )

        for item in all_chunks:

            embedding = create_embedding(
                item["text"]
            )

            conn.execute(
                """
                INSERT INTO chunks
                (filename, page, text, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (
                    item["filename"],
                    item["page"],
                    item["text"],
                    json.dumps(embedding)
                )
            )

        pages = len(
            set(item["page"] for item in all_chunks)
        )

        chunk_count = len(all_chunks)

        conn.execute(
            """
            INSERT INTO documents
            (filename, pages, chunks)
            VALUES (?, ?, ?)
            """,
            (
                filename,
                pages,
                chunk_count
            )
        )

        conn.commit()
        conn.close()

        return {
            "filename": filename,
            "pages": pages,
            "chunks": chunk_count
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"PDF processing error: {str(e)}"
        )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0

    return float(
        np.dot(a, b) / denominator
    )


# ============================================================
# EXPERIMENT NUMBER DETECTION
# ============================================================

def detect_experiment_number(question):

    patterns = [
        r"experiment\s*(?:no\.?|number)?\s*(\d+)",
        r"exp\s*(?:no\.?)?\s*(\d+)",
        r"experiment\s*#\s*(\d+)"
    ]

    question_lower = question.lower()

    for pattern in patterns:

        match = re.search(
            pattern,
            question_lower
        )

        if match:
            return int(match.group(1))

    return None


# ============================================================
# EXPERIMENT MATCHING
# ============================================================

def experiment_match(text, experiment_number):

    if experiment_number is None:
        return 0

    text_lower = text.lower()

    number = str(experiment_number)

    patterns = [
        rf"\bexperiment\s*{number}\b",
        rf"\bexperiment\s*no\.?\s*{number}\b",
        rf"\bexperiment\s*number\s*{number}\b",
        rf"\bexp\.?\s*{number}\b",
        rf"\bexp\s*no\.?\s*{number}\b"
    ]

    for pattern in patterns:

        if re.search(pattern, text_lower):
            return 1

    return 0


# ============================================================
# RAG SEARCH
# ============================================================

def search(question, top_k=6):

    query_embedding = create_embedding(
        question
    )

    experiment_number = detect_experiment_number(
        question
    )

    conn = get_db()

    rows = conn.execute(
        """
        SELECT filename, page, text, embedding
        FROM chunks
        ORDER BY page
        """
    ).fetchall()

    conn.close()

    results = []

    for row in rows:

        try:

            embedding = json.loads(
                row["embedding"]
            )

            semantic_score = cosine_similarity(
                query_embedding,
                embedding
            )

            exact_match = experiment_match(
                row["text"],
                experiment_number
            )

            # ------------------------------------------------
            # Hybrid score
            # ------------------------------------------------

            final_score = semantic_score

            if experiment_number is not None:

                if exact_match:
                    final_score += 0.45

                # Nearby experiment pages get a smaller boost
                # after an exact experiment heading is found.
                if exact_match:
                    final_score += 0.10

            results.append(
                (
                    final_score,
                    {
                        "filename": row["filename"],
                        "page": row["page"],
                        "text": row["text"],
                        "semantic_score": semantic_score,
                        "exact_match": exact_match
                    }
                )
            )

        except Exception:
            continue

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # --------------------------------------------------------
    # If a specific experiment was requested, prioritize
    # exact experiment chunks.
    # --------------------------------------------------------

    if experiment_number is not None:

        exact = [
            r for r in results
            if r[1]["exact_match"] == 1
        ]

        other = [
            r for r in results
            if r[1]["exact_match"] == 0
        ]

        results = exact + other

    return results[:top_k]


# ============================================================
# GEMINI GENERATION WITH RETRY + FALLBACK
# ============================================================

def generate_with_fallback(prompt):

    if not client:

        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is not configured."
        )

    for attempt in range(2):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )

            if response.text:
                return response.text

        except Exception as e:

            error_text = str(e)

            if (
                "503" not in error_text
                and "UNAVAILABLE" not in error_text
            ):

                raise HTTPException(
                    status_code=500,
                    detail=f"Gemini generation error: {error_text}"
                )

            time.sleep(2)

    fallback_models = [
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash-lite"
    ]

    last_error = ""

    for fallback in fallback_models:

        try:

            response = client.models.generate_content(
                model=fallback,
                contents=prompt
            )

            if response.text:
                return response.text

        except Exception as e:

            last_error = str(e)

    raise HTTPException(
        status_code=503,
        detail=(
            "Gemini is temporarily unavailable. "
            "Please try again in a few seconds. "
            f"Last error: {last_error}"
        )
    )


# ============================================================
# RAG GENERATION
# ============================================================

def generate(req, instruction):

    hits = search(req.question)

    if not hits:

        raise HTTPException(
            status_code=400,
            detail=(
                "No lab manual content is available. "
                "Please upload a PDF first."
            )
        )

    context_parts = []

    for score, r in hits:

        context_parts.append(
            f"""
[{r['filename']}, page {r['page']}]
{r['text']}
"""
        )

    context = "\n".join(context_parts)

    experiment_number = detect_experiment_number(
        req.question
    )

    experiment_instruction = ""

    if experiment_number:

        experiment_instruction = f"""
The student is asking specifically about Experiment {experiment_number}.

IMPORTANT:
- First identify the actual Experiment {experiment_number} section
  from the supplied context.
- Do not confuse it with Experiment 2, 3, 4, etc.
- Use the exact experiment title from the manual when available.
- Prefer information from chunks containing
  "Experiment {experiment_number}".
- If the exact section is not present in the retrieved context,
  clearly say that it could not be found.
"""

    prompt = f"""
You are AI StudyMate, an intelligent laboratory manual
study assistant.

{instruction}

{experiment_instruction}

IMPORTANT RULES:

1. Use the supplied laboratory manual context for
   manual-specific facts.

2. Do not invent experiment numbers, titles, procedures,
   algorithms, code or results.

3. If the student asks about an experiment, identify the
   correct experiment before answering.

4. Give clear and exam-friendly explanations.

5. If code is requested, provide concise working code.

6. If the manual does not contain enough information,
   clearly say so.

7. Organize answers with headings and bullet points
   whenever useful.

8. Mention page numbers when they are useful.

MANUAL CONTEXT:
{context}

STUDENT QUESTION:
{req.question}
"""

    answer = generate_with_fallback(
        prompt
    )

    return answer, hits


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI StudyMate API",
        "docs": "/docs"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    conn = get_db()

    document_count = conn.execute(
        "SELECT COUNT(*) FROM documents"
    ).fetchone()[0]

    chunk_count = conn.execute(
        "SELECT COUNT(*) FROM chunks"
    ).fetchone()[0]

    conn.close()

    return {
        "status": "ok",
        "gemini_configured": bool(client),
        "documents": document_count,
        "chunks": chunk_count,
        "model": MODEL
    }


# ============================================================
# DOCUMENT LIST
# ============================================================

@app.get("/documents")
def documents():

    conn = get_db()

    rows = conn.execute(
        """
        SELECT filename, pages, chunks
        FROM documents
        ORDER BY filename
        """
    ).fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# PDF UPLOAD
# ============================================================

@app.post("/upload")
async def upload(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="File name missing."
        )

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    pdf_bytes = await file.read()

    result = ingest_pdf(
        file.filename,
        pdf_bytes
    )

    return {
        "message": "PDF uploaded and indexed successfully.",
        **result
    }


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
def chat(req: Req):

    if not client:

        raise HTTPException(
            status_code=500,
            detail="Configure GOOGLE_API_KEY first."
        )

    if not req.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question required."
        )

    answer, hits = generate(
        req,
        "Answer the student's question."
    )

    sources = []

    seen = set()

    for score, r in hits:

        key = (
            r["filename"],
            r["page"]
        )

        if key not in seen:

            sources.append({
                "filename": r["filename"],
                "page": r["page"],
                "score": round(score, 3)
            })

            seen.add(key)

    return {
        "answer": answer,
        "sources": sources
    }


# ============================================================
# VIVA
# ============================================================

@app.post("/viva")
def viva(req: Req):

    if not client:

        raise HTTPException(
            status_code=500,
            detail="Configure GOOGLE_API_KEY first."
        )

    answer, _ = generate(
        req,
        """
Create 10 important laboratory viva questions
with short answers.

Mix:

- Basic questions
- Conceptual questions
- Implementation questions
- Output questions
- Important exam questions
"""
    )

    return {
        "answer": answer
    }


# ============================================================
# STUDY PLAN
# ============================================================

@app.post("/study-plan")
def study_plan(req: Req):

    if not client:

        raise HTTPException(
            status_code=500,
            detail="Configure GOOGLE_API_KEY first."
        )

    answer, _ = generate(
        req,
        """
Create a concise laboratory examination preparation plan.

Include:

- Important experiments
- Important concepts
- Coding practice
- Viva preparation
- Common mistakes
- Final revision checklist
"""
    )

    return {
        "answer": answer
    }