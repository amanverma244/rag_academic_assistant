import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.ingest import PDFIngestor
from src.retriever import VectorStoreManager
from src.generator import RAGPipeline

# Page Configuration
st.set_page_config(
    page_title="Semantica - Academic Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a modern look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    .css-1v0mbdj {
        border-radius: 10px;
    }
    .source-box {
        background-color: #1e2530;
        border-left: 4px solid #4f46e5;
        padding: 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-top: 10px;
        color: #d1d5db;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize RAG components
@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()

try:
    rag = get_rag_pipeline()
except Exception as e:
    st.error(f"Error initializing RAG pipeline. Check your API key. Details: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/graduation-cap.png", width=60)
    st.title("Semantica")
    st.markdown("Your intelligent research assistant powered by dense retrieval and Gemini.")
    
    st.divider()
    st.header("📄 Document Vault")
    
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        os.makedirs("data/raw_pdfs", exist_ok=True)
        file_path = os.path.join("data/raw_pdfs", uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        if st.button("🚀 Process & Index Paper", use_container_width=True):
            with st.status("Processing document pipeline...", expanded=True) as status:
                try:
                    st.write("Extracting text and metadata...")
                    ingestor = PDFIngestor(file_path)
                    pages = ingestor.extract_text_with_metadata()
                    
                    st.write("Chunking text semantically...")
                    chunks = ingestor.chunk_documents(pages)
                    
                    st.write("Embedding & storing in ChromaDB...")
                    vector_manager = VectorStoreManager()
                    vector_manager.add_documents(chunks)
                    
                    status.update(label="Indexing complete!", state="complete", expanded=False)
                    st.success(f"Indexed **{uploaded_file.name}** ({len(chunks)} chunks)")
                except Exception as e:
                    status.update(label="Indexing failed", state="error", expanded=True)
                    st.error(f"Error: {e}")

    st.divider()
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - Upload papers related to your study domain.
    - Ask specific questions about methodology, results, or architecture.
    - Check the citation expander below each answer for source tracking.
    """)

# --- MAIN CHAT INTERFACE ---
st.title("🎓 Interactive Research Paper Assistant")
st.markdown("Chat with your academic library, extract deep insights, and reference exact page numbers instantly.")

# Initialize chat history state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history container
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there are sources stored with the message, display them neatly
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Source Citations"):
                for src in message["sources"]:
                    st.markdown(f"- **{src['source']}** (Page {src['page']})")

# User chat input
if query := st.chat_input("Ask a question about your papers (e.g., What is the core architecture?)..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 *Synthesizing answer from papers...*")
        
        try:
            result = rag.answer_query(query, top_k=3)
            answer = result["answer"]
            sources = result["sources"]
            
            # Deduplicate sources for display
            unique_sources = []
            seen = set()
            for meta in sources:
                identifier = (meta['source'], meta['page'])
                if identifier not in seen:
                    seen.add(identifier)
                    unique_sources.append(meta)

            message_placeholder.markdown(answer)
            
            # Show interactive source expander
            if unique_sources:
                with st.expander("📚 View Source Citations"):
                    for src in unique_sources:
                        st.markdown(f"- **{src['source']}** (Page {src['page']})")

            # Save to history state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": unique_sources
            })
            
        except Exception as e:
            error_msg = f"⚠️ An error occurred during generation: {e}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})