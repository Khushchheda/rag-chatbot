from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read PDF
reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

# Split text into chunks
chunks = text.split(".")

# Remove empty chunks
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

print(f"\nTotal Chunks: {len(chunks)}")

# Generate embeddings
chunk_embeddings = model.encode(chunks)

print("\nPDF Semantic Search Ready!\n")

while True:
    query = input("Ask a question: ")

    if query.lower() == "exit":
        break

    # Convert query into embedding
    query_embedding = model.encode([query])

    # Calculate similarities
    similarities = cosine_similarity(
        query_embedding,
        chunk_embeddings
    )[0]

    # Get top result
    best_match_index = np.argmax(similarities)
    top_indices = np.argsort(similarities)[-3:][::-1]

    print("\nMost Relevant Chunk:\n")
    print(chunks[best_match_index])

    print(f"\nSimilarity Score: {similarities[best_match_index]:.4f}\n")

    print("Top 3 Relevant Chunks:\n")
    for i, idx in enumerate(top_indices, 1):
        print(f"{i}. {chunks[idx]}")
        print(f"   Similarity Score: {similarities[idx]:.4f}\n")

        