import os
import fitz
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PDF_PATH = "CV_David_Stocco.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        text= text.strip()
        if text:
            pages.append((page_num + 1, text))
    doc.close()
    print(f"Extracted text from {len(pages)} pages")
    return pages

def chunk_text(text, chunk_size, overlap): 
    chunks = []
    start = 0 
    while start < len(text):
        end = start + chunk_size 
        chunk = text[start:end]
        chunks.append(chunk)  

        start += chunk_size - overlap 

    return chunks

def embed_text(texts):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
    )
    
    return [e.values for e in response.embeddings]

def index_pdf(pdf_path):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    try: 
        chroma_client.delete_collection("pdf_rag")
    except:
        pass
    
    
    collection = chroma_client.create_collection("pdf_rag")

    pages = extract_text_from_pdf(pdf_path)

    all_chunks = []
    all_metadata = []
    all_ids = []
    chunk_index = 0

    for page_num, page_text in pages:
        
        chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)

        
        for chunk in chunks:
            if len(chunk.strip()) < 30:
                continue

            all_chunks.append(chunk)
            all_metadata.append({
                "source": os.path.basename(pdf_path),
                "page": page_num,
                "chunk": chunk_index,
            })
            all_ids.append(f"chunk_{chunk_index}")
            chunk_index += 1 

    print(f"Generando embeddings en lotes para {len(all_chunks)} fragmentos...")
    
    # Procesamiento por lotes (Batching) DENTRO de la función
    BATCH_SIZE = 100
    all_embeddings = []

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        batch_embeddings = embed_text(batch)
        all_embeddings.extend(batch_embeddings)
        print(f" -> Embedded batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

    
    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        metadatas=all_metadata,
        ids=all_ids
    )

    print(f"¡Listo! Se indexaron {len(all_chunks)} fragmentos en ChromaDB.")
    return collection

def ask(collection, question, n_results=3):
    question_embedding = embed_text(question)[0]

    
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    context = "\n\n".join(chunks)
    
    
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
Always mention the page the answer came from if possible.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""


    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    
    print(f"\n--- PREGUNTA: {question} ---")
    print(f"Respuesta: {response.text}")
    print("\nFuentes recuperadas:")
    for meta, dist in zip(metadatas, distances):
        print(f" -> Página {meta['page']} del archivo {meta['source']} (Distancia: {dist:.4f})")

if __name__ == "__main__": 
    collection = index_pdf(PDF_PATH)
    ask(collection, "what project has David worked on")
    ask(collection, "what skills does David have")










