"""
Emergency Streamlit App for Ethical AI RAG System
Uses simplified vector store without PyTorch dependencies
Compatible with Python 3.13 and Streamlit Cloud deployment issues
"""

import streamlit as st
import os
import sys
import logging
from pathlib import Path
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to Python path for local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Emergency environment setup
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Global variables for module loading status
MODULES_LOADED = False
IMPORT_ERRORS = []

def safe_import():
    """Safely import required modules with detailed error reporting."""
    global MODULES_LOADED, IMPORT_ERRORS
    
    try:
        # Import basic modules first
        import PyPDF2
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Import our simplified modules
        from pdf_extractor import PDFExtractor
        from document_processor import DocumentProcessor
        from vector_store_simple import VectorStore
        
        MODULES_LOADED = True
        logger.info("✅ All modules imported successfully")
        return True
        
    except ImportError as e:
        error_msg = f"Import Error: {str(e)}"
        IMPORT_ERRORS.append(error_msg)
        logger.error(error_msg)
        return False
    except Exception as e:
        error_msg = f"Unexpected Error: {str(e)}"
        IMPORT_ERRORS.append(error_msg)
        logger.error(error_msg)
        return False

class EmergencyRAGSystem:
    """Emergency RAG system using simplified components."""
    
    def __init__(self):
        """Initialize the emergency RAG system."""
        self.pdf_extractor = None
        self.document_processor = None
        self.vector_store = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            if not MODULES_LOADED:
                return False
            
            # Import here to avoid issues if modules aren't loaded
            from pdf_extractor import PDFExtractor
            from document_processor import DocumentProcessor
            from vector_store_simple import VectorStore
            
            self.pdf_extractor = PDFExtractor()
            self.document_processor = DocumentProcessor()
            self.vector_store = VectorStore()
            
            self.is_initialized = True
            logger.info("✅ Emergency RAG system initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG system: {e}")
            return False
    
    def process_pdf(self, pdf_file) -> bool:
        """Process a PDF file and add to vector store."""
        try:
            if not self.is_initialized:
                return False
            
            # Extract text from PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                text = self.pdf_extractor.extract_text(tmp_file_path)
                if not text or len(text.strip()) < 50:
                    st.error("❌ Could not extract meaningful text from PDF")
                    return False
                
                # Process into chunks
                chunks = self.document_processor.process_text(text)
                if not chunks:
                    st.error("❌ Could not process text into chunks")
                    return False
                
                # Add to vector store
                success = self.vector_store.add_documents(chunks)
                if success:
                    st.success(f"✅ Successfully processed PDF into {len(chunks)} chunks")
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
            st.error(f"❌ Error processing PDF: {str(e)}")
            logger.error(f"PDF processing error: {e}")
            return False
    
    def query(self, question: str, num_results: int = 3):
        """Query the RAG system."""
        try:
            if not self.is_initialized:
                return {"error": "System not initialized"}
            
            if not question.strip():
                return {"error": "Empty question"}
            
            # Search for relevant documents
            results = self.vector_store.similarity_search(question, k=num_results)
            
            if not results:
                return {
                    "answer": "I couldn't find any relevant information in the uploaded documents.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Simple answer generation (concatenate top results)
            relevant_texts = [doc for doc, score, metadata in results if score > 0.1]
            
            if not relevant_texts:
                return {
                    "answer": "I found some potentially relevant information, but the similarity scores were too low to provide a confident answer.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Create a simple answer by combining relevant chunks
            answer = f"Based on the uploaded documents:\n\n"
            for i, text in enumerate(relevant_texts[:3], 1):
                # Truncate very long texts
                display_text = text[:500] + "..." if len(text) > 500 else text
                answer += f"{i}. {display_text}\n\n"
            
            return {
                "answer": answer,
                "sources": relevant_texts,
                "confidence": results[0][1] if results else 0.0,
                "num_sources": len(relevant_texts)
            }
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {"error": f"Query failed: {str(e)}"}

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Emergency Ethical AI RAG",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🚨 Emergency Ethical AI RAG System")
    st.markdown("**Simplified deployment-compatible version**")
    
    # Emergency notice
    st.warning("""
    ⚠️ **Emergency Deployment Mode**
    
    This is a simplified version designed to work around deployment issues:
    - ✅ No PyTorch dependencies (avoids meta tensor errors)
    - ✅ Python 3.13 compatible  
    - ✅ Uses TF-IDF embeddings instead of transformer models
    - ✅ Simplified but functional RAG capabilities
    """)
    
    # Check module loading status
    if not MODULES_LOADED:
        st.error("❌ **Module Loading Failed**")
        st.error("Some required modules could not be imported.")
        
        if IMPORT_ERRORS:
            st.error("Import Errors:")
            for error in IMPORT_ERRORS:
                st.code(error)
        
        st.info("💡 Try using requirements-emergency.txt for deployment")
        st.stop()
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = EmergencyRAGSystem()
        st.session_state.documents_processed = 0
        st.session_state.query_history = []
    
    # Initialize RAG system
    if not st.session_state.rag_system.is_initialized:
        with st.spinner("Initializing Emergency RAG System..."):
            success = st.session_state.rag_system.initialize()
            if not success:
                st.error("❌ Failed to initialize RAG system")
                st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # System stats
        if st.session_state.rag_system.vector_store:
            stats = st.session_state.rag_system.vector_store.get_stats()
            st.metric("Documents Processed", stats.get('total_documents', 0))
            st.metric("Vector Dimensions", stats.get('embedding_dimension', 0))
        
        st.header("🔧 Emergency Controls")
        if st.button("Clear All Data"):
            if st.session_state.rag_system.vector_store:
                st.session_state.rag_system.vector_store.clear()
                st.session_state.documents_processed = 0
                st.success("✅ Data cleared")
                st.rerun()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF files to build your knowledge base"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"Process {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        success = st.session_state.rag_system.process_pdf(uploaded_file)
                        if success:
                            st.session_state.documents_processed += 1
                            st.rerun()
    
    with col2:
        st.header("❓ Ask Questions")
        
        if st.session_state.documents_processed == 0:
            st.info("📝 Please upload and process some PDF documents first")
        else:
            question = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="What would you like to know about the uploaded documents?"
            )
            
            if st.button("🔍 Search", type="primary") and question:
                with st.spinner("Searching documents..."):
                    result = st.session_state.rag_system.query(question, 3)
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        # Display answer
                        st.subheader("📋 Answer")
                        st.write(result["answer"])
                        
                        # Display metadata
                        st.write(f"**Confidence:** {result.get('confidence', 0):.2f}")
                        st.write(f"**Sources Used:** {result.get('num_sources', 0)}")

# Initialize modules on import
safe_import()

if __name__ == "__main__":
    main() 
