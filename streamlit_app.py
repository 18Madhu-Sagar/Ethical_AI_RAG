import streamlit as st
import os
import tempfile
import shutil
from typing import List, Dict
import time

# Import our RAG system components
from rag_system import EthicalAIRAG
from pdf_extractor import PDFExtractor
from document_processor import DocumentProcessor
from vector_store import VectorStore
from response_refiner import ResponseRefiner

# Page configuration
st.set_page_config(
    page_title="Ethical AI RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .source-tag {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
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

def save_uploaded_files(uploaded_files) -> str:
    """Save uploaded files to a temporary directory."""
    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir)
    
    temp_dir = tempfile.mkdtemp()
    st.session_state.temp_dir = temp_dir
    
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    return temp_dir

def setup_rag_system(pdf_directory: str, use_refinement: bool = True) -> bool:
    """Set up the RAG system with uploaded PDFs."""
    try:
        # Initialize RAG system
        st.session_state.rag_system = EthicalAIRAG(
            pdf_directory=pdf_directory,
            vector_db_path=os.path.join(pdf_directory, "chroma_db"),
            use_refinement=use_refinement
        )
        
        # Setup the system
        success = st.session_state.rag_system.setup(force_rebuild=True)
        st.session_state.system_ready = success
        
        return success
        
    except Exception as e:
        st.error(f"Failed to setup RAG system: {str(e)}")
        return False

def display_system_stats():
    """Display system statistics in the sidebar."""
    if st.session_state.system_ready and st.session_state.rag_system:
        status = st.session_state.rag_system.get_system_status()
        
        st.sidebar.markdown("### 📊 System Statistics")
        st.sidebar.metric("PDF Files", status.get('pdf_files', 0))
        st.sidebar.metric("Document Chunks", status.get('total_chunks', 0))
        
        if 'chunk_stats' in status:
            chunk_stats = status['chunk_stats']
            st.sidebar.metric("Average Chunk Length", f"{chunk_stats['avg_chunk_length']:.0f}")
            
            st.sidebar.markdown("### 📚 Sources")
            for source, count in chunk_stats['chunks_per_source'].items():
                st.sidebar.write(f"📄 **{source}**: {count} chunks")

def display_query_interface():
    """Display the main query interface."""
    st.markdown('<div class="query-section">', unsafe_allow_html=True)
    
    # Query input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Ask a question about AI ethics:",
            placeholder="e.g., What are the main ethical principles of AI?",
            key="query_input"
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
    if (query_button or query) and query.strip():
        process_query(query.strip(), query_type, response_length)
    
    st.markdown('</div>', unsafe_allow_html=True)

def process_query(query: str, query_type: str, response_length: str = "medium"):
    """Process the user query and display results."""
    if not st.session_state.system_ready:
        st.error("Please upload and process PDF files first!")
        return
    
    # Add to query history
    st.session_state.query_history.append({
        'query': query,
        'type': query_type,
        'timestamp': time.strftime("%H:%M:%S")
    })
    
    with st.spinner(f"Processing your {query_type.lower()}..."):
        try:
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
            st.error(f"Error processing query: {str(e)}")

def display_simple_results(query: str, results: List):
    """Display simple question results."""
    st.markdown(f"### 🔍 Results for: *{query}*")
    
    if not results:
        st.warning("No relevant documents found.")
        return
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'unknown')
        chunk_id = doc.metadata.get('chunk_id', 'N/A')
        
        with st.expander(f"📄 Result {i} from {source} (Chunk {chunk_id})"):
            st.markdown(f'<div class="result-box">{doc.page_content}</div>', unsafe_allow_html=True)

def display_comprehensive_answer(query: str, answer: str):
    """Display comprehensive answer."""
    st.markdown(f"### 💡 Comprehensive Answer: *{query}*")
    
    if answer:
        st.markdown(f'<div class="result-box">{answer}</div>', unsafe_allow_html=True)
        
        # Show supporting evidence
        with st.expander("📚 View Supporting Evidence"):
            results = st.session_state.rag_system.ask_ethics_question(query, num_results=3, refine_response=False)
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get('source', 'unknown')
                st.markdown(f"**Source {i}: {source}**")
                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                st.markdown("---")
    else:
        st.warning("No comprehensive answer could be generated.")

def display_source_comparison(query: str, sources: Dict):
    """Display source comparison results."""
    st.markdown(f"### 📊 Source Comparison: *{query}*")
    
    if not sources:
        st.warning("No sources found for comparison.")
        return
    
    # Create tabs for each source
    if len(sources) > 1:
        tabs = st.tabs([f"📄 {source}" for source in sources.keys()])
        
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
        st.warning("No relevant documents found.")
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
    if st.session_state.query_history:
        st.sidebar.markdown("### 📝 Query History")
        
        for i, entry in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            with st.sidebar.expander(f"{entry['timestamp']} - {entry['type']}"):
                st.write(f"**Query:** {entry['query']}")

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Ethical AI RAG System</h1>', unsafe_allow_html=True)
    st.markdown("Upload PDF documents about AI ethics and ask questions to get intelligent, source-backed answers.")
    
    # Sidebar
    st.sidebar.markdown("## 📁 Document Upload")
    
    # File upload
    uploaded_files = st.sidebar.file_uploader(
        "Upload AI Ethics PDF Documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload multiple PDF files containing AI ethics content"
    )
    
    # Processing options
    st.sidebar.markdown("### ⚙️ Processing Options")
    use_refinement = st.sidebar.checkbox("Enable Response Refinement", value=True, 
                                       help="Use AI to refine and improve responses")
    
    # Process uploaded files
    if uploaded_files:
        if st.sidebar.button("🚀 Process Documents", use_container_width=True):
            with st.spinner("Processing uploaded documents..."):
                # Save files to temporary directory
                temp_dir = save_uploaded_files(uploaded_files)
                
                # Setup RAG system
                if setup_rag_system(temp_dir, use_refinement):
                    st.success(f"✅ Successfully processed {len(uploaded_files)} documents!")
                    st.session_state.uploaded_files = [f.name for f in uploaded_files]
                else:
                    st.error("❌ Failed to process documents. Please check your files and try again.")
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.sidebar.markdown("### 📄 Uploaded Files")
        for filename in st.session_state.uploaded_files:
            st.sidebar.write(f"✓ {filename}")
    
    # Display system stats
    display_system_stats()
    
    # Display query history
    display_query_history()
    
    # Main content area
    if st.session_state.system_ready:
        # Query interface
        display_query_interface()
        
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
        for i, (col, query) in enumerate(zip(cols, example_queries)):
            with col:
                if st.button(f"💡 Example {i+1}", key=f"example_{i}", use_container_width=True):
                    st.session_state.query_input = query
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