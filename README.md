# AI StudyMate

A polished Generative AI lab-manual analyzer for CREOZEN-style RAG requirements.

## Stack
React + Vite, FastAPI, Google Gemini, PyMuPDF, NumPy, SQLite local vector store.

## RAG
PDF -> extraction -> chunks -> Gemini embeddings -> local SQLite vector store -> cosine retrieval -> Gemini generation -> filename/page sources.

## Credentials
Only `GOOGLE_API_KEY` is required. No Qdrant, Pinecone, Supabase, OpenAI, MongoDB Atlas or Firebase.

## Local run
Backend:
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn main:app --reload --port 8000
```
Frontend:
```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```
Keep the Gemini key only in `backend/.env`.

## Production
Deploy frontend separately from backend. Set `VITE_API_URL` to the hosted backend. Use persistent storage for `backend/data` on the backend host.
