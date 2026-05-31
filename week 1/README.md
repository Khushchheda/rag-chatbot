# RAG Chatbot

## Overview
A Retrieval-Augmented Generation (RAG) chatbot that answers questions from PDF documents using semantic search and vector retrieval.

## Features
- PDF ingestion
- Text chunking
- Embedding generation
- ChromaDB vector storage
- Semantic retrieval
- Context-aware answer generation

## Tech Stack
- Python
- Sentence Transformers
- ChromaDB
- Hugging Face Transformers
- PyPDF

## Architecture

User Query
↓
Embedding Model
↓
ChromaDB Retrieval
↓
Relevant Chunks
↓
LLM
↓
Answer

## Future Improvements
- Streamlit UI
- PDF Upload Interface
- Conversation Memory
- Source Citations
- RAGAS Evaluation
- Hugging Face Deployment

# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using:

- LangChain
- ChromaDB
- HuggingFace Embeddings
- Cohere Reranking
- Streamlit

Upload a PDF and ask questions about its content.