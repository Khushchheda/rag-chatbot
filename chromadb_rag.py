from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client
client = chromadb.Client()

# Create collection
collection = client.create_collection("pdf_collection")

# Read PDF
reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

# Split into chunks
chunks = text.split(".")

chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

print(f"\nTotal Chunks: {len(chunks)}")

# Store chunks in ChromaDB
for i, chunk in enumerate(chunks):

    embedding = model.encode(chunk).tolist()

    collection.add(
        metadatas=[{"source": "sample.pdf"}],
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk]
    )

print("\nChunks stored in ChromaDB!")

# Query loop
while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print("\nTop Results:\n")

    for doc in results['documents'][0]:
        print("-", doc)
        print()