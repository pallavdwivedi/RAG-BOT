# RAG-BOT
RAG-Powered Article Knowledge Chatbot
-Tech Stack: Python, Streamlit, LangChain, ChromaDB, Groq API, HuggingFace
-Domain: Information Retrieval + Natural Language Generation
-Deployment: Local development with potential cloud scaling


User Interface (Streamlit)
    ↓
Article Ingestion (WebBaseLoader + BeautifulSoup)
    ↓
Text Processing (RecursiveCharacterTextSplitter)
    ↓
Embeddings Generation (HuggingFace Transformers)
    ↓
Vector Storage (ChromaDB with persistence)
    ↓
Retrieval System (Similarity Search)
    ↓
LLM Generation (Groq API - Llama 3.3 70B)
    ↓
Response Delivery


Data Flow
-Ingestion: URL → WebBaseLoader → Raw HTML → Clean Text

-Processing: Text → Chunks → Embeddings → Vector Store

-Retrieval: Query → Query Embedding → Similarity Search → Top-K Documents

-Generation: Context + Query → LLM → Grounded Response
