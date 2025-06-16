import streamlit as st
import os
import sys
import tempfile
import shutil
from typing import List, Dict, Optional
import time
import traceback
from pathlib import Path

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import our RAG system components with error handling
try:
    from rag_system import EthicalAIRAG
    from pdf_extractor import PDFExtractor
    from document_processor import DocumentProcessor
    from vector_store import VectorStore
    from response_refiner import ResponseRefiner
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"❌ Failed to import required modules: {e}")
    st.error("Please ensure all required files are present in the deployment.")
    MODULES_LOADED = False

# Page configuration
st.set_page_config(
    page_title="Ethical AI RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Environment setup for deployment
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TRANSFORMERS_CACHE', '/tmp/transformers_cache')

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .query-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff4e6;
        padding: 1rem;
        border-left: 4px solid #ff8800;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #e6ffe6;
        padding: 1rem;
        border-left: 4px solid #44aa44;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def safe_cleanup():
    """Safely cleanup temporary files and directories."""
    try:
        if 'temp_dir' in st.session_state and st.session_state.temp_dir:
            if os.path.exists(st.session_state.temp_dir):
                shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        
        # Clear large objects from session state
        keys_to_clear = ['rag_system', 'temp_dir', 'large_results']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
                
    except Exception as e:
        st.warning(f"Cleanup warning: {e}")

def initialize_session_state():
    """Initialize session state variables with error handling."""
    try:
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'system_ready' not in st.session_state:
            st.session_state.system_ready = False
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'temp_dir' not in st.session_state:
            st.session_state.temp_dir = None
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'processing_error' not in st.session_state:
            st.session_state.processing_error = None
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(time.time())
    except Exception as e:
        st.error(f"Session initialization error: {e}")

def save_uploaded_files(uploaded_files) -> Optional[str]:
    """Save uploaded files to a temporary directory with robust error handling."""
    try:
        # Clean up any existing temp directory
        if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        
        # Create new temporary directory
        temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
        st.session_state.temp_dir = temp_dir
        
        saved_files = []
        for uploaded_file in uploaded_files:
            try:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(uploaded_file.name)
            except Exception as e:
                st.error(f"Failed to save {uploaded_file.name}: {e}")
                continue
        
        if saved_files:
            st.success(f"✅ Saved {len(saved_files)} files: {', '.join(saved_files)}")
            return temp_dir
        else:
            st.error("❌ Failed to save any files")
            return None
            
    except Exception as e:
        st.error(f"File saving error: {e}")
        return None

def setup_rag_system(pdf_directory: str, use_refinement: bool = True) -> bool:
    """Set up the RAG system with comprehensive error handling."""
    if not MODULES_LOADED:
        st.error("❌ Cannot setup RAG system - modules not loaded properly")
        return False
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Initialize components
        status_text.text("🔧 Initializing RAG system components...")
        progress_bar.progress(10)
        
        # Check if directory exists and has PDF files
        if not os.path.exists(pdf_directory):
            st.error(f"❌ Directory not found: {pdf_directory}")
            return False
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        if not pdf_files:
            st.error(f"❌ No PDF files found in {pdf_directory}")
            return False
        
        st.info(f"📄 Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")
        
        # Step 2: Initialize RAG system
        status_text.text("🚀 Creating RAG system...")
        progress_bar.progress(20)
        
        vector_db_path = os.path.join(pdf_directory, "chroma_db")
        
        try:
            st.session_state.rag_system = EthicalAIRAG(
                pdf_directory=pdf_directory,
                vector_db_path=vector_db_path,
                use_refinement=use_refinement
            )
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {e}")
            st.error("This might be due to missing dependencies or insufficient memory.")
            return False
        
        # Step 3: Setup the system
        status_text.text("📚 Processing documents and building vector store...")
        progress_bar.progress(40)
        
        try:
            with st.spinner("Processing documents... This may take a few minutes."):
                success = st.session_state.rag_system.setup(force_rebuild=True)
        except Exception as e:
            st.error(f"❌ Document processing failed: {e}")
            st.error("Error details:")
            st.code(traceback.format_exc())
            return False
        
        progress_bar.progress(90)
        
        if success:
            st.session_state.system_ready = True
            status_text.text("✅ RAG system ready!")
            progress_bar.progress(100)
            
            # Display setup summary
            if hasattr(st.session_state.rag_system, 'stats'):
                stats = st.session_state.rag_system.stats
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"""
                **📊 Setup Complete!**
                - PDF files processed: {stats.get('pdf_files', 0)}
                - Document chunks created: {stats.get('total_chunks', 0)}
                - Average chunk length: {stats.get('chunk_stats', {}).get('avg_chunk_length', 0):.0f} chars
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            return True
        else:
            status_text.text("❌ Setup failed")
            st.error("❌ Failed to setup RAG system. Check the error messages above.")
            return False
        
    except Exception as e:
        progress_bar.progress(0)
        status_text.text("❌ Setup failed with error")
        st.error(f"❌ Setup error: {e}")
        st.error("Full error traceback:")
        st.code(traceback.format_exc())
        return False

def validate_query(query: str) -> bool:
    """Validate user query."""
    if not query or not query.strip():
        st.warning("⚠️ Please enter a question.")
        return False
    
    if len(query) < 3:
        st.warning("⚠️ Query too short. Please enter at least 3 characters.")
        return False
    
    if len(query) > 500:
        st.warning("⚠️ Query too long. Please limit to 500 characters.")
        return False
    
    return True

def safe_process_query(query: str, query_type: str, response_length: str = "medium"):
    """Process query with comprehensive error handling."""
    if not st.session_state.system_ready or not st.session_state.rag_system:
        st.error("❌ RAG system not ready. Please upload and process PDF files first.")
        return
    
    if not validate_query(query):
        return
    
    # Add to query history
    st.session_state.query_history.append({
        'query': query,
        'type': query_type,
        'timestamp': time.strftime("%H:%M:%S")
    })
    
    # Keep only last 20 queries
    if len(st.session_state.query_history) > 20:
        st.session_state.query_history = st.session_state.query_history[-20:]
    
    try:
        with st.spinner(f"Processing your {query_type.lower()}..."):
            if query_type == "Simple Question":
                results = st.session_state.rag_system.ask_ethics_question(query, num_results=3)
                display_simple_results(query, results)
                
            elif query_type == "Comprehensive Answer":
                answer = st.session_state.rag_system.get_comprehensive_answer(query, response_length)
                display_comprehensive_answer(query, answer)
                
            elif query_type == "Compare Sources":
                sources = st.session_state.rag_system.compare_sources(query, num_results=6)
                display_source_comparison(query, sources)
                
            elif query_type == "Keyword Search":
                results = st.session_state.rag_system.search_keywords(query, num_results=5)
                display_keyword_results(query, results)
                
    except Exception as e:
        st.error(f"❌ Error processing query: {e}")
        st.error("This might be due to:")
        st.markdown("""
        - Memory limitations in the deployment environment
        - Network connectivity issues
        - Temporary service unavailability
        
        **Try:**
        - Using a shorter, simpler query
        - Refreshing the page and trying again
        - Checking if your documents were processed correctly
        """)
        
        if st.checkbox("Show detailed error information"):
            st.code(traceback.format_exc())

def display_system_stats():
    """Display system statistics in the sidebar with error handling."""
    try:
        if st.session_state.system_ready and st.session_state.rag_system:
            status = st.session_state.rag_system.get_system_status()
            
            st.sidebar.markdown("### 📊 System Statistics")
            st.sidebar.metric("PDF Files", status.get('pdf_files', 0))
            st.sidebar.metric("Document Chunks", status.get('total_chunks', 0))
            
            if 'chunk_stats' in status:
                chunk_stats = status['chunk_stats']
                avg_length = chunk_stats.get('avg_chunk_length', 0)
                st.sidebar.metric("Average Chunk Length", f"{avg_length:.0f}")
                
                st.sidebar.markdown("### 📚 Sources")
                chunks_per_source = chunk_stats.get('chunks_per_source', {})
                for source, count in chunks_per_source.items():
                    # Truncate long filenames
                    display_name = source if len(source) <= 25 else source[:22] + "..."
                    st.sidebar.write(f"📄 **{display_name}**: {count} chunks")
    except Exception as e:
        st.sidebar.error(f"Stats error: {e}")

def display_simple_results(query: str, results: List):
    """Display simple question results."""
    st.markdown(f"### 🔍 Results for: *{query}*")
    
    if not results:
        st.markdown('<div class="warning-box">⚠️ No relevant documents found. Try rephrasing your question or using different keywords.</div>', unsafe_allow_html=True)
        return
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'unknown')
        chunk_id = doc.metadata.get('chunk_id', 'N/A')
        
        with st.expander(f"📄 Result {i} from {source} (Chunk {chunk_id})"):
            st.markdown(f'<div class="result-box">{doc.page_content}</div>', unsafe_allow_html=True)

def display_comprehensive_answer(query: str, answer: str):
    """Display comprehensive answer."""
    st.markdown(f"### 💡 Comprehensive Answer: *{query}*")
    
    if answer and answer.strip():
        st.markdown(f'<div class="result-box">{answer}</div>', unsafe_allow_html=True)
        
        # Show supporting evidence
        with st.expander("📚 View Supporting Evidence"):
            try:
                results = st.session_state.rag_system.ask_ethics_question(query, num_results=3, refine_response=False)
                if results:
                    for i, doc in enumerate(results, 1):
                        source = doc.metadata.get('source', 'unknown')
                        st.markdown(f"**Source {i}: {source}**")
                        preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                        st.text(preview)
                        st.markdown("---")
                else:
                    st.info("No supporting evidence found.")
            except Exception as e:
                st.error(f"Could not load supporting evidence: {e}")
    else:
        st.markdown('<div class="warning-box">⚠️ No comprehensive answer could be generated. Try rephrasing your question.</div>', unsafe_allow_html=True)

def display_source_comparison(query: str, sources: Dict):
    """Display source comparison results."""
    st.markdown(f"### 📊 Source Comparison: *{query}*")
    
    if not sources:
        st.markdown('<div class="warning-box">⚠️ No sources found for comparison. Try using different keywords.</div>', unsafe_allow_html=True)
        return
    
    # Create tabs for each source
    if len(sources) > 1:
        tab_names = [f"📄 {source[:20]}..." if len(source) > 20 else f"📄 {source}" for source in sources.keys()]
        tabs = st.tabs(tab_names)
        
        for tab, (source, docs) in zip(tabs, sources.items()):
            with tab:
                st.markdown(f"**{len(docs)} relevant chunks from {source}**")
                for i, doc in enumerate(docs, 1):
                    with st.expander(f"Excerpt {i}"):
                        st.write(doc.page_content)
    else:
        # Single source
        source, docs = next(iter(sources.items()))
        st.markdown(f"**{len(docs)} relevant chunks from {source}**")
        for i, doc in enumerate(docs, 1):
            with st.expander(f"Excerpt {i}"):
                st.write(doc.page_content)

def display_keyword_results(query: str, results: List):
    """Display keyword search results with scores."""
    st.markdown(f"### 🔎 Keyword Search: *{query}*")
    
    if not results:
        st.markdown('<div class="warning-box">⚠️ No relevant documents found. Try different keywords.</div>', unsafe_allow_html=True)
        return
    
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get('source', 'unknown')
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Result {i}** from *{source}*")
        with col2:
            st.metric("Relevance", f"{score:.3f}")
        
        preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        st.markdown(f'<div class="result-box">{preview}</div>', unsafe_allow_html=True)
        
        with st.expander("View Full Content"):
            st.write(doc.page_content)

def display_query_history():
    """Display query history in sidebar."""
    try:
        if st.session_state.query_history:
            st.sidebar.markdown("### 📝 Query History")
            
            # Show only last 5 queries
            recent_queries = list(reversed(st.session_state.query_history[-5:]))
            for i, entry in enumerate(recent_queries, 1):
                with st.sidebar.expander(f"{entry['timestamp']} - {entry['type']}"):
                    st.write(f"**Query:** {entry['query']}")
    except Exception as e:
        st.sidebar.error(f"History error: {e}")

def main():
    """Main Streamlit application with enhanced error handling."""
    
    # Check if modules loaded correctly
    if not MODULES_LOADED:
        st.error("❌ Critical Error: Required modules could not be loaded.")
        st.error("Please check that all Python files are present and dependencies are installed.")
        st.stop()
    
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Ethical AI RAG System</h1>', unsafe_allow_html=True)
    st.markdown("Upload PDF documents about AI ethics and ask questions to get intelligent, source-backed answers.")
    
    # Sidebar
    st.sidebar.markdown("## 📁 Document Upload")
    
    # File upload with validation
    uploaded_files = st.sidebar.file_uploader(
        "Upload AI Ethics PDF Documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload multiple PDF files containing AI ethics content (max 200MB total)"
    )
    
    # File validation
    if uploaded_files:
        total_size = sum(f.size for f in uploaded_files)
        max_size = 200 * 1024 * 1024  # 200MB
        
        if total_size > max_size:
            st.sidebar.error(f"❌ Total file size ({total_size/1024/1024:.1f}MB) exceeds 200MB limit")
            uploaded_files = None
        else:
            st.sidebar.success(f"✅ {len(uploaded_files)} files ready ({total_size/1024/1024:.1f}MB)")
    
    # Processing options
    st.sidebar.markdown("### ⚙️ Processing Options")
    use_refinement = st.sidebar.checkbox(
        "Enable Response Refinement", 
        value=False,  # Disabled by default for stability
        help="Use AI to refine and improve responses (may be slower)"
    )
    
    # Process uploaded files
    if uploaded_files:
        if st.sidebar.button("🚀 Process Documents", use_container_width=True):
            with st.spinner("Processing uploaded documents..."):
                # Save files to temporary directory
                temp_dir = save_uploaded_files(uploaded_files)
                
                if temp_dir:
                    # Setup RAG system
                    if setup_rag_system(temp_dir, use_refinement):
                        st.session_state.uploaded_files = [f.name for f in uploaded_files]
                        st.balloons()
                    else:
                        st.error("❌ Failed to process documents. Please check your files and try again.")
                        if st.checkbox("Show troubleshooting tips"):
                            st.markdown("""
                            **Common issues:**
                            - PDF files are image-based (scanned documents) - text extraction may fail
                            - Files are password protected
                            - Insufficient memory for processing large documents
                            - Network connectivity issues
                            
                            **Try:**
                            - Using smaller PDF files
                            - Converting scanned PDFs to text-based PDFs
                            - Refreshing the page and trying again
                            """)
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.sidebar.markdown("### 📄 Uploaded Files")
        for filename in st.session_state.uploaded_files:
            # Truncate long filenames for display
            display_name = filename if len(filename) <= 30 else filename[:27] + "..."
            st.sidebar.write(f"✓ {display_name}")
    
    # Display system stats
    display_system_stats()
    
    # Display query history
    display_query_history()
    
    # Add cleanup button in sidebar
    if st.sidebar.button("🧹 Clear Session"):
        safe_cleanup()
        st.experimental_rerun()
    
    # Main content area
    if st.session_state.system_ready:
        # Query interface
        st.markdown('<div class="query-section">', unsafe_allow_html=True)
        
        # Query input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Ask a question about AI ethics:",
                placeholder="e.g., What are the main ethical principles of AI?",
                key="query_input",
                max_chars=500
            )
        
        with col2:
            query_button = st.button("🔍 Search", use_container_width=True)
        
        # Query type selection
        query_type = st.selectbox(
            "Query Type:",
            ["Simple Question", "Comprehensive Answer", "Compare Sources", "Keyword Search"],
            help="Choose how you want to process your query"
        )
        
        # Response length for comprehensive answers
        if query_type == "Comprehensive Answer":
            response_length = st.selectbox(
                "Response Length:",
                ["short", "medium", "long"],
                index=1
            )
        else:
            response_length = "medium"
        
        # Process query
        if (query_button or query) and query and query.strip():
            safe_process_query(query.strip(), query_type, response_length)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Demo queries
        st.markdown("### 🎯 Try These Example Queries")
        example_queries = [
            "What are the main ethical principles of AI?",
            "How can we prevent algorithmic bias?", 
            "What are the privacy concerns with AI systems?",
            "How should AI be governed and regulated?",
            "What is the role of transparency in AI?"
        ]
        
        cols = st.columns(len(example_queries))
        for i, (col, query_ex) in enumerate(zip(cols, example_queries)):
            with col:
                if st.button(f"💡 Example {i+1}", key=f"example_{i}", use_container_width=True):
                    st.session_state.query_input = query_ex
                    st.experimental_rerun()
    
    else:
        # Welcome message
        st.markdown("""
        ### 👋 Welcome to the Ethical AI RAG System!
        
        This system helps you explore and understand AI ethics by allowing you to:
        
        - 📄 **Upload multiple PDF documents** about AI ethics
        - 🔍 **Ask natural language questions** about the content
        - 💡 **Get comprehensive answers** backed by source citations
        - 📊 **Compare perspectives** from different sources
        - 🔎 **Search for specific keywords** with relevance scoring
        
        **To get started:**
        1. Upload one or more PDF files using the sidebar
        2. Click "Process Documents" to analyze the content
        3. Start asking questions about AI ethics!
        
        **Supported document types:** Academic papers, policy documents, ethics guidelines, research reports, and more.
        """)
        
        # Sample upload area
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 🚀 Ready to Start?")
        st.markdown("Upload your AI ethics documents using the **Document Upload** section in the sidebar.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        Powered by LangChain, Chroma, and Streamlit | 
        <a href='https://github.com' target='_blank'>View Source Code</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
