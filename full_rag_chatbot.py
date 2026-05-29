from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb

# -----------------------------
# Load models
# -----------------------------

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# -----------------------------
# Initialize ChromaDB
# -----------------------------

client = chromadb.Client()

collection = client.create_collection("rag_collection")

# -----------------------------
# Read PDF
# -----------------------------

reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

# -----------------------------
# Chunking
# -----------------------------

chunks = text.split(".")

chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

print(f"\nTotal Chunks: {len(chunks)}")

# -----------------------------
# Store embeddings
# -----------------------------

for i, chunk in enumerate(chunks):

    embedding = embedding_model.encode(chunk).tolist()

    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk],
        metadatas=[{"source": "sample.pdf"}]
    )

print("\nRAG Chatbot Ready!")

# -----------------------------
# Chat Loop
# -----------------------------

while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    # Embed query
    query_embedding = embedding_model.encode(query).tolist()

    # Retrieve top chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_chunks = results['documents'][0]

    context = "\n".join(retrieved_chunks)

    # Prompt for LLM
    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    # Generate answer
    response = generator(
        prompt,
        max_new_tokens=150
    )

    answer = response[0]['generated_text']

    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== SOURCES ===\n")

    for chunk in retrieved_chunks:
        print("-", chunk[:200])
        print()