import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

sentences = [
    "The dog ran across the park.",
    "A puppy sprinted through the garden.",  # Similar to #1
    "The quarterly earnings report is due.",   # Very different
]

embeddings = []

for sentence in sentences:
    response = client.models.embed_content(
        model="gemini-embedding-001",  # Usamos el modelo que descubrimos
        contents=sentence,
    )
    # Extraemos los valores del embedding
    embedding = response.embeddings[0].values
    embeddings.append(embedding)
    
    print(f"Sentence : {sentence}")
    print(f"Embedding: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ...]")
    print(f"Length   : {len(embedding)} numbers\n")

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

sim_1_2 = dot_product(embeddings[0], embeddings[1])
sim_1_3 = dot_product(embeddings[0], embeddings[2])

print("-" * 50)
print(f"Similarity: sentence 1 vs sentence 2 -> {sim_1_2:.4f} (similar meaning)")
print(f"Similarity: sentence 1 vs sentence 3 -> {sim_1_3:.4f} (different meaning)")
print()
print("Notice: similar sentences score HIGHER. That's embeddings working.")