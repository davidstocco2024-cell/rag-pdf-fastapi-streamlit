import os 
import fitz
from google import genai
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import chromadb

load_dotenv()

client = genai.Client()

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_or_create_collection():
    try:
        return chroma_client.get_collection("rag_collection")
    except:
        return chroma_client.create_collection("rag_collection")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50 

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages

def chunk_text(text):
    chunks = []
    start = 0 
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks 

def embed_text(texts):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )
    return [e.values for e in response.embeddings]

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    collection = get_or_create_collection()

    pages = extract_text(tmp_path)
    all_chunks = []
    all_metadata = []
    all_ids = []
    chunk_index = collection.count()

    for page in pages:
        page_num = page["page"]
        page_text = page["text"]
        for chunk in chunk_text(page_text):
            if len(chunk.strip()) < 50:
                continue
            all_chunks.append(chunk)
            all_metadata.append({
                "source": file.filename,
                "page": page_num
            })
            all_ids.append(f"chunk_{chunk_index}")
            chunk_index += 1

    all_embeddings = []
    for i in range(0, len(all_chunks), 100): 
        batch = all_chunks[i: i + 100]
        all_embeddings.extend(embed_text(batch))

    # Store in ChromaDB
    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        metadatas=all_metadata,
        ids=all_ids,
    )

    os.unlink(tmp_path)

    return {
        "message": f"Ingested {file.filename} successfully", 
        "chunks_added": len(all_chunks),
        "total_chunks": collection.count(),
    }

class AskRequest(BaseModel):
    question: str
    n_results: int = 3

@app.post("/ask")
async def ask(request: AskRequest):
    collection = get_or_create_collection()

    if collection.count() == 0:
        return {"error": "No documents ingested yet"}

    question_embedding = embed_text([request.question])[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=request.n_results,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []
    sources = []

    for chunk, meta, dist in zip(chunks, metadatas, distances):
        context_parts.append(chunk)
        sources.append({
            "source": meta["source"],
            "page": meta["page"],
            "score": round(dist, 4),
        })

    context = "\n\n--- \n\n".join(context_parts)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
Be concise and specific. Always mention the source document and page if relevant.
If the answer is not in the context, say "i dont know based on the provided documents."

Context:
{context}

Question: {request.question}
Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return {
        "question": request.question,
        "answer": response.text,
        "sources": sources,
    }

@app.get("/")
def root():
    collection = get_or_create_collection()
    return {
        "status": "running",
        "total_chunks": collection.count(),
    }
