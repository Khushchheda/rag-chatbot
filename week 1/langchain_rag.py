import os

os.system('cls')
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline
import cohere
from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(
    os.getenv("COHERE_API_KEY")
)

chat_history = []

loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

print(f"Total chunks: {len(docs)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}
)

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

print("\nRAG Chatbot Ready!")
print("Type 'exit' to quit.\n")

history_text = ""

while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break
    
    
    results = retriever.invoke(query)
    print(f"\nRetrieved Chunks: {len(results)}")

    documents = []
    metadata_map = []

    for doc in results:
        documents.append(doc.page_content)
        metadata_map.append(doc.metadata)

    rerank_results = co.rerank(
    query=query,
    documents=documents,
    top_n=3,
    model="rerank-v3.5"
    )

    top_chunks = []
    top_sources = []

    for result in rerank_results.results:
        top_chunks.append(
            documents[result.index]
    )
        top_sources.append(
            metadata_map[result.index]
        )

    print("\n=== RERANKED CHUNKS ===")

    for i, chunk in enumerate(top_chunks, start=1):
        print(f"\nChunk {i}:")
        print(chunk[:250])

    context = "\n".join(top_chunks)

    history_text = "\n".join(
        [f"User: {item['question']}\nAssistant: {item['answer']}" 
         for item in chat_history]
    )

    prompt = f"""
    Use the conversation history and context to answer.

    Conversation History:
    {history_text}

    Context:
    {context}

    Current Question:
    {query}

    Answer:
    """

    response = generator(
        prompt,
        max_new_tokens=150
    )

    answer = response[0]["generated_text"]

    chat_history.append(
        {
            "question": query,
            "answer": answer
        }
    )

    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== SOURCES ===\n")    

    for source in top_sources:

        page_number = source["page"] + 1

        print(
            f"{source['source']} | Page {page_number}"
        )
        