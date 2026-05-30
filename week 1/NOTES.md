What is an embedding?
An embedding is a dense numerical vector representation of data (such as text, images, or audio) that captures its semantic meaning. Similar pieces of data have embeddings that are close together in vector space, enabling semantic search and similarity comparison.

What is cosine similarity?
Cosine similarity is a metric used to measure how similar two vectors are by calculating the cosine of the angle between them. A value close to 1 indicates high similarity, while a value close to 0 indicates little or no similarity.

Why use ChromaDB?
ChromaDB is a vector database used to store, index, and retrieve embeddings efficiently. It enables fast semantic search by finding vectors that are most similar to a query embedding, making it useful for RAG applications and AI-powered search systems.

What is RAG?
RAG stands for Retrieval-Augmented Generation. 
RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with language generation. Documents are split into chunks, converted into embeddings, and stored in a vector database. When a user asks a question, relevant chunks are retrieved using semantic search and provided as context to a language model, which generates a grounded answer.

Difference between RAG and fine-tuning?
| RAG                                          | Fine-Tuning                                  |
| -------------------------------------------- | -------------------------------------------- |
| Retrieves external information at query time | Modifies the model's weights during training |
| Knowledge can be updated without retraining  | Requires retraining to learn new information |
| Faster and cheaper to update                 | More expensive and time-consuming            |
| Best for document Q&A and dynamic knowledge  | Best for changing model behavior or style    |
| Uses vector databases and retrieval systems  | Uses training datasets and optimization      |

LangChain
- PyPDFLoader
- RecursiveCharacterTextSplitter
- HuggingFaceEmbeddings
- ChromaDB
- Retriever

RAG Improvements
- Conversation Memory
- Cohere Reranking
- Source Citations

Streamlit
- file_uploader()
- text_input()
- Building a web UI for RAG

FastAPI
- FastAPI()
- GET endpoints
- POST endpoints
- Pydantic BaseModel
- Swagger Docs