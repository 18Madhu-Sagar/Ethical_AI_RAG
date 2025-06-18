"""
Simplified Streamlit App for Ethical AI RAG System
Deployment-ready version with minimal dependencies
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

# Session state initialization
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

def safe_import_rag():
    """Safely import RAG system with error handling."""
    try:
        # Only import when actually needed
        from rag_system import AdvancedRAGSystem
        return AdvancedRAGSystem, None
    except ImportError as e:
        return None, f"Import error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"

def initialize_rag_system():
    """Initialize the RAG system with error handling."""
    if st.session_state.system_initialized and st.session_state.rag_system:
        return st.session_state.rag_system, None
    
    try:
        RAGSystem, error = safe_import_rag()
        if error:
            return None, error
        
        # Initialize with minimal configuration - NO MODEL LOADING
        rag_system = RAGSystem(
            pdf_directory=".",
            vector_db_path="./vector_db",
            embedding_model="tfidf",
            llm_provider="enhanced_simple",  # Use simple generation to avoid complex dependencies
            llm_model="simple"
        )
        
        # Don't initialize LLM during startup
        rag_system.llm_pipeline = None
        rag_system.tokenizer = None
        
        st.session_state.rag_system = rag_system
        st.session_state.system_initialized = True
        logger.info("✅ RAG system initialized successfully (minimal mode)")
        return rag_system, None
        
    except Exception as e:
        error_msg = f"Failed to initialize RAG system: {e}"
        logger.error(error_msg)
        return None, error_msg

def process_uploaded_file(uploaded_file) -> bool:
    """Process an uploaded PDF file."""
    try:
        rag_system, error = initialize_rag_system()
        if error:
            st.error(f"❌ {error}")
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
                rag_system.is_ready = True
                st.session_state.documents_processed = True
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
    """Ask a question using the RAG system."""
    try:
        rag_system, error = initialize_rag_system()
        if error:
            return {"error": error}
        
        if not rag_system.is_ready or not st.session_state.documents_processed:
            return {"error": "No documents processed yet. Please upload a PDF first."}
        
        # Get answer from RAG system
        result = rag_system.ask_question(question)
        return result
        
    except Exception as e:
        logger.error(f"Question processing error: {e}")
        return {"error": f"Failed to process question: {str(e)}"}

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Ethical AI RAG System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🤖 Ethical AI RAG System")
    st.markdown("**Document Analysis with AI-Powered Question Answering**")
    
    # System info
    st.info("""
    🚀 **Features:**
    - ✅ PDF document processing
    - ✅ TF-IDF embeddings for semantic search
    - ✅ AI-powered question answering
    - ✅ Document chunking and analysis
    - ✅ Deployment-ready architecture
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Initialize system button
        if st.button("🔄 Initialize System"):
            rag_system, error = initialize_rag_system()
            if error:
                st.error(f"❌ {error}")
            else:
                st.success("✅ System initialized successfully!")
        
        # Status display
        if st.session_state.system_initialized:
            st.success("✅ System Ready")
            if st.session_state.documents_processed:
                st.success("✅ Documents Processed")
            else:
                st.warning("⚠️ No documents processed")
        else:
            st.warning("⚠️ System not initialized")
        
        st.header("🔧 Controls")
        if st.button("🗑️ Clear All Data"):
            st.session_state.rag_system = None
            st.session_state.system_initialized = False
            st.session_state.documents_processed = False
            st.success("✅ All data cleared")
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
        st.markdown("Query your documents using AI analysis")
        
        question = st.text_area(
            "Enter your question:",
            placeholder="e.g., What are the main ethical principles for AI development?",
            height=100
        )
        
        if st.button("🔍 Get Answer") and question.strip():
            with st.spinner("Generating answer..."):
                result = ask_question(question.strip())
                
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.success("✅ Answer generated successfully!")
                    
                    # Display answer
                    st.subheader("📋 Answer")
                    st.write(result.get("answer", "No answer generated"))
                    
                    # Display metadata
                    if result.get('confidence'):
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    
                    # Display sources
                    if result.get('sources'):
                        st.subheader("📚 Sources")
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(f"Source {i}"):
                                st.write(source.get('content', 'No content'))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Ethical AI RAG System** - Powered by TF-IDF and AI
    
    🔗 **How it works:**
    1. Upload PDF documents
    2. System processes and chunks the text
    3. Ask questions about the content
    4. Get AI-powered answers with source references
    """)

if __name__ == "__main__":
    main() 