from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge base
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks are inspired by the human brain.",
    "Cricket is a popular sport in India.",
    "Transformers are used in large language models.",
    "Virat Kohli is a famous Indian cricketer.",
]

# Convert documents into embeddings
document_embeddings = model.encode(documents)

print("\nSemantic Search Engine Ready!\n")

while True:
    query = input("Enter your query: ")

    if query.lower() == "exit":
        break

    # Convert query into embedding
    query_embedding = model.encode([query])

    # Calculate similarity
    similarities = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    # Find best matches (sorted by relevance)
    sorted_indices = np.argsort(similarities)[::-1]

    print("\nTop Relevant Results:")
    for i, idx in enumerate(sorted_indices[:3], 1):
        print(f"{i}. {documents[idx]}")
        print(f"   Similarity Score: {similarities[idx]:.4f}\n")