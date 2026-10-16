import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

documents = [
    "The return policy allows customers to return items within 30 days of purchase.",
    "All returns must include the original receipt and packaging.",
    "Refunds are processed within 5 to 7 business days after the item is received.",
    "Items that are damaged or used cannot be returned unless they are defective.",
    "To start a return, visit our website and fill out the return request form.",
]

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_first_rag")

print("Embedding and storing documents...")

for i, doc in enumerate(documents):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=doc,
    )
    embedding = response.embeddings[0].values
    
    collection.add(
        documents=[doc],
        embeddings=[embedding],
        ids=[f"doc_{i}"]
    )

print(f"Stored {len(documents)} documents in ChromaDB\n")

# — STEP 4: Ask a question —
question = "How long do I have to return something?"

# 1. Convertimos la pregunta en un embedding usando Gemini
q_response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question,
)
question_embedding = q_response.embeddings[0].values

# 2. Consultamos a ChromaDB pidiendo los 2 mejores resultados
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)

retrieved_chunks = results["documents"][0]
print("Retrieved chunks:")
for chunk in retrieved_chunks:
    print(f" -> {chunk}")
print()

# — STEP 5: Build context and generate answer —
context = "\n".join(retrieved_chunks)

prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""

# 3. Generamos la respuesta final utilizando Gemini (adaptado del chat.completions del video)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
)

answer = response.text

# 4. Imprimimos el resultado final al estilo del tutorial
print(f"Question: {question}")
print(f"Answer: {answer}")

