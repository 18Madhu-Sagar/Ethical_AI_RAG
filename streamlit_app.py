"""
Advanced Streamlit App for Ethical AI RAG System
Uses proper LLM integration with LangChain for high-quality answers
"""

import streamlit as st
import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Environment setup
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Global variables - use session state instead of global variables
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

def initialize_rag_system():
    """Initialize the advanced RAG system with lazy loading."""
    if st.session_state.system_initialized and st.session_state.rag_system:
        return st.session_state.rag_system
    
    try:
        from rag_system import AdvancedRAGSystem
        
        # Initialize with auto-detection (HuggingFace if available, enhanced simple otherwise)
        rag_system = AdvancedRAGSystem(
            pdf_directory=".",
            vector_db_path="./simple_vector_db",
            embedding_model="tfidf",
            llm_provider="auto",
            llm_model="microsoft/DialoGPT-medium"
        )
        
        st.session_state.rag_system = rag_system
        st.session_state.system_initialized = True
        logger.info("✅ Enhanced RAG system with auto-detected LLM initialized")
        return rag_system
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        st.error(f"Failed to initialize RAG system: {e}")
        return None

def process_uploaded_file(uploaded_file) -> bool:
    """Process an uploaded PDF file."""
    try:
        rag_system = initialize_rag_system()
        if not rag_system:
            return False
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Extract text
            text = rag_system.pdf_extractor.extract_text_robust(tmp_file_path)
            if not text or len(text.strip()) < 50:
                st.error("❌ Could not extract meaningful text from PDF")
                return False
            
            # Process into documents
            temp_doc = {uploaded_file.name: text}
            rag_documents = rag_system.document_processor.create_rag_documents(temp_doc)
            
            if not rag_documents:
                st.error("❌ Could not process text into chunks")
                return False
            
            # Add to vector store
            success = rag_system.vector_store.create_vectorstore(rag_documents)
            if success:
                # Create QA chain
                rag_system._create_qa_chain()
                rag_system.is_ready = True
                
                st.success(f"✅ Successfully processed {uploaded_file.name} into {len(rag_documents)} chunks")
                return True
            else:
                st.error("❌ Failed to add documents to vector store")
                return False
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        logger.error(f"File processing error: {e}")
        return False

def ask_question(question: str) -> Dict[str, Any]:
    """Ask a question using the advanced RAG system."""
    try:
        rag_system = initialize_rag_system()
        if not rag_system:
            return {"error": "RAG system not initialized"}
        
        if not rag_system.is_ready:
            return {"error": "No documents processed yet. Please upload a PDF first."}
        
        # Get answer from advanced RAG system
        result = rag_system.ask_question(question)
        return result
        
    except Exception as e:
        logger.error(f"Question processing error: {e}")
        return {"error": f"Failed to process question: {str(e)}"}

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Advanced Ethical AI RAG",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🤖 Enhanced Ethical AI RAG System")
    st.markdown("**RAG system with smart LLM detection that works with Python 3.13**")
    
    # System info
    st.info("""
    🚀 **Enhanced Features with Smart LLM Detection**
    - ✅ TF-IDF embeddings (fast and reliable)
    - ✅ Auto-detects best available LLM (HuggingFace or Enhanced Simple)
    - ✅ Document chunking and processing
    - ✅ Semantic search with cosine similarity
    - ✅ AI-powered answer synthesis with question analysis
    - ✅ Works with Python 3.13
    """)
    
    # Initialize system only when needed
    rag_system = None
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Only initialize when sidebar is accessed
        if st.button("🔄 Initialize System") or st.session_state.system_initialized:
            rag_system = initialize_rag_system()
            
        if rag_system:
            status = rag_system.get_system_status()
            st.metric("System Ready", "✅ Yes" if status['ready'] else "❌ No")
            st.metric("Documents Loaded", "✅ Yes" if status['documents_loaded'] else "❌ No")
            st.metric("LLM Provider", status['llm_provider'])
            st.metric("LLM Model", status['llm_model'])
            st.metric("Embedding Model", status['embedding_model'])
            
            vector_stats = status['vector_store_stats']
            st.metric("Vector Documents", vector_stats['num_documents'])
            st.metric("Embedding Dimension", vector_stats['embedding_dimension'])
        else:
            st.warning("⚠️ System not initialized. Click 'Initialize System' to start.")
        
        st.header("🔧 Controls")
        if st.button("🗑️ Clear Vector Store"):
            if rag_system and rag_system.vector_store:
                rag_system.vector_store.clear()
                rag_system.is_ready = False
                st.success("✅ Vector store cleared")
                st.rerun()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Upload")
        st.markdown("Upload PDF documents for analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload PDF documents to analyze with the RAG system"
        )
        
        if uploaded_file is not None:
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size:,} bytes")
            
            if st.button("🔄 Process Document"):
                with st.spinner("Processing document..."):
                    success = process_uploaded_file(uploaded_file)
                    if success:
                        st.rerun()
    
    with col2:
        st.header("❓ Ask Questions")
        st.markdown("Query your documents using smart AI analysis")
        
        question = st.text_area(
            "Enter your question:",
            placeholder="e.g., What are the main ethical principles for AI development?",
            height=100
        )
        
        if st.button("🔍 Get Answer") and question.strip():
            with st.spinner("Generating AI-powered answer..."):
                result = ask_question(question.strip())
                
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.success("✅ Answer generated successfully!")
                    
                    # Display answer
                    st.subheader("📋 Answer")
                    st.write(result.get("answer", "No answer generated"))
                    
                    # Display metadata
                    col_conf, col_sources, col_model = st.columns(3)
                    with col_conf:
                        st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
                    with col_sources:
                        st.metric("Sources Used", result.get('num_sources', 0))
                    with col_model:
                        st.metric("LLM Model", result.get('llm_model', 'Unknown'))
                    
                    # Display sources
                    if result.get('sources'):
                        st.subheader("📚 Sources")
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(f"Source {i} - {source.get('metadata', {}).get('source', 'Unknown')}"):
                                st.write(source.get('content', 'No content'))
                                if source.get('metadata'):
                                    st.json(source['metadata'])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Simplified Ethical AI RAG System** - Powered by TF-IDF and Scikit-learn
    
    🔗 **Architecture:**
    - PDF Processing → Document Chunking → TF-IDF Embeddings → Multi-source Answers
    - Uses reliable TF-IDF for semantic understanding
    - Simplified but effective RAG implementation
    """)

if __name__ == "__main__":
    main() 