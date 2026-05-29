from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "cricket",
    "movies",
    "AI",
    "food",
]

embeddings = model.encode(sentences)

print("\n=== Cosine Similarity Scores ===\n")

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        similarity = cosine_similarity(
            [embeddings[i]],
            [embeddings[j]]
        )[0][0]

        print(f'"{sentences[i]}"')
        print(f'"{sentences[j]}"')
        print(f"Similarity Score: {similarity:.4f}\n")