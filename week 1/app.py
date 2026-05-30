import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline

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
        search_kwargs={"k": 10}
    )

    st.success("Vector Database Created")

    generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
    )

    question = st.text_input("Ask a question")

    if question:

        results = retriever.invoke(question)

        context = "\n".join(
            [doc.page_content for doc in results[:3]]
        )

        prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

        response = generator(
            prompt,
            max_new_tokens=150
        )

        answer = response[0]["generated_text"]

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Sources")

        for doc in results[:3]:

            st.write(
                f"Page {doc.metadata['page'] + 1}"
        )