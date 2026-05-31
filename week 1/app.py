import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline
import os
import cohere
from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(
    os.getenv("COHERE_API_KEY")
)

st.title("📄 RAG Chatbot")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(
            uploaded_file.getvalue()
        )

        pdf_path = tmp_file.name

    st.success("PDF Uploaded Successfully")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    st.write(
        f"Total Chunks: {len(docs)}"
    )

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 20}
    )

    st.success("Vector Database Created")

    generator = pipeline(
    "text-generation",
    model="google/flan-t5-large"    )

    question = st.text_input("Ask a question")

    if question:

        results = retriever.invoke(question)

        documents = []
        metadata_map = []

        for doc in results:
            documents.append(doc.page_content)
            metadata_map.append(doc.metadata)

        rerank_results = co.rerank(
            query=question,
            documents=documents,
            top_n=5,
            model="rerank-v3.5"
        )

        top_chunks = []
        top_sources = []

        seen_chunks = set()

        for result in rerank_results.results:

            chunk = documents[result.index]

            if chunk not in seen_chunks:

                seen_chunks.add(chunk)

                top_chunks.append(chunk)

                top_sources.append(
                    metadata_map[result.index]
                )

        for result in rerank_results.results:

            top_chunks.append(
                documents[result.index]
            )

            top_sources.append(
                metadata_map[result.index]
            )

        context = "\n".join(top_chunks)

        prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
        st.subheader("Reranked Chunks")

        for i, chunk in enumerate(top_chunks, start=1):
            st.write(f"Chunk {i}")
            st.write(chunk[:500])
        response = generator(
            prompt,
            max_new_tokens=150
        )

        answer = response[0]["generated_text"]

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Sources")

        for source in top_sources:

            st.write(
                f"Page {source['page'] + 1}"
            )