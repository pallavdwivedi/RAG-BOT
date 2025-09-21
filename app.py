import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

# Load environment variables
load_dotenv()

# Constants
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "articles" not in st.session_state:
    st.session_state.articles = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )
if "messages" not in st.session_state:
    st.session_state.messages = []
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")

def add_article(url):
    """Add article from URL to vector store"""
    try:
        # Load and process document
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        if not docs:
            return None, "No content loaded from URL."
        
        # Extract title
        title = docs[0].metadata.get('title', docs[0].page_content[:100] + "...")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(docs)
        
        # Add metadata
        for split in splits:
            split.metadata['source_url'] = url
            split.metadata['title'] = title
        
        # Store in vector DB
        st.session_state.vectorstore.add_documents(splits)
        
        # Add to article list
        if not any(a['url'] == url for a in st.session_state.articles):
            st.session_state.articles.append({'url': url, 'title': title})
        
        return title, None
        
    except Exception as e:
        return None, str(e)

def setup_rag_chain():
    """Setup RAG chain with LLM and retriever"""
    if not st.session_state.groq_api_key:
        return None, "Please add your Groq API key to the .env file."
    
    try:
        # Initialize LLM
        llm = ChatGroq(
            groq_api_key=st.session_state.groq_api_key,
            model_name=GROQ_MODEL,
            temperature=0.1
        )
        
        # Setup retriever
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Get prompt template
        prompt = hub.pull("rlm/rag-prompt")
        
        # Format documents function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, None
        
    except Exception as e:
        return None, f"Error setting up RAG chain: {str(e)}"

# Sidebar
with st.sidebar:
    st.title("🤖 RAG Chatbot Settings")
    
    # API Key status
    if st.session_state.groq_api_key:
        st.success("✅ API Key Loaded")
        st.text(f"Key: {st.session_state.groq_api_key[:20]}...")
    else:
        st.error("❌ No API Key Found")
        st.info("Add GROQ_API_KEY to your .env file")
    
    st.divider()
    
    # Vector store info
    try:
        collection = st.session_state.vectorstore._collection
        doc_count = collection.count()
        st.info(f"📚 Documents in DB: {doc_count}")
    except:
        st.info("📚 Documents in DB: 0")
    
    # Articles list
    with st.expander("📖 Added Articles", expanded=True):
        if not st.session_state.articles:
            st.write("No articles added yet.")
        else:
            for i, article in enumerate(st.session_state.articles):
                st.write(f"**{i+1}.** [{article['title'][:40]}...]({article['url']})")
    
    # 🔍 ADD THE DEBUG BUTTON HERE:
    if st.button("🔍 Show What's Stored"):
        if st.session_state.articles:
            # Show first article's chunks
            try:
                retriever = st.session_state.vectorstore.as_retriever()
                test_docs = retriever.get_relevant_documents("content")[:3]
                
                st.write("**Sample stored content:**")
                for i, doc in enumerate(test_docs):
                    st.write(f"**Chunk {i+1}:** {doc.page_content[:200]}...")
                    st.write(f"**Source:** {doc.metadata.get('title', 'Unknown')}")
                    st.write("---")
            except Exception as e:
                st.error(f"Debug error: {e}")
        else:
            st.write("No articles to debug.")
    
    st.divider()
    
    # Clear data
    if st.button("🗑️ Clear All Data", type="secondary"):
        if st.session_state.articles:
            st.session_state.articles = []
            st.session_state.messages = []
            # Clear vector store
            try:
                st.session_state.vectorstore.delete_collection()
                st.session_state.vectorstore = Chroma(
                    persist_directory=PERSIST_DIRECTORY,
                    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                )
                st.success("All data cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing data: {e}")


# Main content
st.title("🔍 RAG Chatbot with Article Knowledge")
st.markdown("**Add articles from URLs and chat with your personalized knowledge base!**")

# Article input section
st.subheader("📝 Add Articles")
col1, col2 = st.columns([4, 1])

with col1:
    url_input = st.text_input(
        "Article URL:",
        placeholder="https://www.example.com/article",
        help="Paste any article URL to add it to your knowledge base"
    )

with col2:
    st.write("")
    st.write("")
    add_button = st.button("➕ Add Article", type="primary", use_container_width=True)

if add_button and url_input:
    if not st.session_state.groq_api_key:
        st.error("❌ Please add your Groq API key to the .env file and restart the app!")
    else:
        with st.spinner("🔄 Processing article... This may take a moment."):
            title, error = add_article(url_input.strip())
            if error:
                st.error(f"❌ Error adding article: {error}")
            else:
                st.success(f"✅ Successfully added: **{title}**")
                st.rerun()

# Chat interface
st.subheader("💬 Chat with your Knowledge Base")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about the articles you've added..."):
    if not st.session_state.articles:
        st.warning("⚠️ Please add some articles first before asking questions!")
    elif not st.session_state.groq_api_key:
        st.error("❌ API key not found. Please check your .env file.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                rag_chain, error = setup_rag_chain()
                if error:
                    response = f"❌ {error}"
                else:
                    try:
                        response = rag_chain.invoke(prompt)
                    except Exception as e:
                        response = f"❌ Error generating response: {str(e)}"
                
                st.markdown(response)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("💡 **Tip**: Add multiple articles for richer conversations!")
with col2:
    st.markdown("🔄 **Models**: Powered by Groq's fast LPU inference")
with col3:
    st.markdown("🔒 **Privacy**: All data stored locally on your machine")

