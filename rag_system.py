from typing import List, Dict, Optional, Tuple
from langchain.schema import Document

from pdf_extractor import PDFExtractor
from document_processor import DocumentProcessor
from vector_store import VectorStore
from response_refiner import ResponseRefiner


class EthicalAIRAG:
    """Complete Ethical AI RAG system combining all components."""
    
    def __init__(self, 
                 pdf_directory: str = ".",
                 vector_db_path: str = "./chroma_db",
                 use_refinement: bool = True,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Ethical AI RAG system.
        
        Args:
            pdf_directory: Directory containing PDF files
            vector_db_path: Path to store vector database
            use_refinement: Whether to use response refinement
            embedding_model: Name of the embedding model to use
        """
        self.pdf_directory = pdf_directory
        self.vector_db_path = vector_db_path
        self.use_refinement = use_refinement
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore(persist_directory=vector_db_path, model_name=embedding_model)
        
        if self.use_refinement:
            self.response_refiner = ResponseRefiner(use_summarizer=False)  # Set to True for AI summarization
        else:
            self.response_refiner = None
        
        # System state
        self.is_ready = False
        self.documents_loaded = False
        self.stats = {}
    
    def setup(self, force_rebuild: bool = False) -> bool:
        """Set up the RAG system by processing PDFs and creating vector store."""
        print("\n" + "="*60)
        print("🚀 SETTING UP ETHICAL AI RAG SYSTEM")
        print("="*60)
        
        try:
            # Step 1: Extract text from PDFs
            print("\n📄 Step 1: Extracting text from PDFs...")
            documents_text = self.pdf_extractor.extract_from_directory(self.pdf_directory)
            
            if not documents_text:
                print("❌ No documents extracted. Setup failed.")
                return False
            
            # Step 2: Process documents into chunks
            print("\n🔧 Step 2: Processing documents into chunks...")
            rag_documents = self.document_processor.create_rag_documents(documents_text)
            
            if not rag_documents:
                print("❌ No document chunks created. Setup failed.")
                return False
            
            # Step 3: Create vector store
            print("\n🧠 Step 3: Creating vector store...")
            success = self.vector_store.create_vectorstore(rag_documents, force_recreate=force_rebuild)
            
            if not success:
                print("❌ Vector store creation failed. Setup failed.")
                return False
            
            # Update system state
            self.is_ready = True
            self.documents_loaded = True
            
            # Store statistics
            self.stats = {
                'pdf_files': len(documents_text),
                'total_chunks': len(rag_documents),
                'chunk_stats': self.document_processor.get_chunk_statistics(rag_documents),
                'vector_info': self.vector_store.get_collection_info()
            }
            
            print("\n" + "="*60)
            print("✅ RAG SYSTEM SETUP COMPLETE")
            print("="*60)
            self._print_setup_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Setup failed with error: {e}")
            return False
    
    def _print_setup_summary(self):
        """Print a summary of the setup process."""
        print(f"📊 SETUP SUMMARY:")
        print(f"   📄 PDF files processed: {self.stats['pdf_files']}")
        print(f"   📋 Document chunks created: {self.stats['total_chunks']}")
        print(f"   🧠 Vector store status: {self.stats['vector_info']['status']}")
        print(f"   📏 Average chunk length: {self.stats['chunk_stats']['avg_chunk_length']:.1f} characters")
        print(f"   🔧 Response refinement: {'Enabled' if self.use_refinement else 'Disabled'}")
    
    def ask_ethics_question(self, question: str, num_results: int = 3, refine_response: bool = True) -> List[Document]:
        """Ask a question about AI ethics and get relevant document chunks."""
        if not self.is_ready:
            print("❌ RAG system not ready. Please run setup() first.")
            return []
        
        print(f"\n🔍 Searching for: '{question}'")
        
        # Perform similarity search
        results = self.vector_store.similarity_search(question, k=num_results)
        
        if not results:
            print("❌ No relevant documents found.")
            return []
        
        print(f"✅ Found {len(results)} relevant chunks")
        
        # Optionally refine responses
        if refine_response and self.response_refiner:
            refined_results = []
            for doc in results:
                refined_content = self.response_refiner.quick_refine(question, doc.page_content, max_words=100)
                refined_doc = Document(
                    page_content=refined_content,
                    metadata=doc.metadata
                )
                refined_results.append(refined_doc)
            return refined_results
        
        return results
    
    def compare_sources(self, topic: str, num_results: int = 4) -> Dict[str, List[Document]]:
        """Compare how different sources address a topic."""
        if not self.is_ready:
            print("❌ RAG system not ready. Please run setup() first.")
            return {}
        
        print(f"\n📊 Comparing sources on: '{topic}'")
        
        # Get relevant documents
        results = self.vector_store.similarity_search(topic, k=num_results)
        
        # Group by source
        source_groups = {}
        for doc in results:
            source = doc.metadata.get('source', 'unknown')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        print(f"✅ Found content from {len(source_groups)} sources")
        for source, docs in source_groups.items():
            print(f"   📄 {source}: {len(docs)} relevant chunks")
        
        return source_groups
    
    def search_keywords(self, keywords: str, num_results: int = 5) -> List[Tuple[Document, float]]:
        """Search for specific keywords with relevance scores."""
        if not self.is_ready:
            print("❌ RAG system not ready. Please run setup() first.")
            return []
        
        print(f"\n🔎 Keyword search: '{keywords}'")
        
        # Use similarity search with scores
        results = self.vector_store.similarity_search_with_score(keywords, k=num_results)
        
        if not results:
            print("❌ No relevant documents found.")
            return []
        
        print(f"✅ Found {len(results)} results with relevance scores")
        
        return results
    
    def get_comprehensive_answer(self, question: str, target_length: str = "medium") -> str:
        """Get a comprehensive answer by combining multiple sources and refining."""
        if not self.is_ready:
            print("❌ RAG system not ready. Please run setup() first.")
            return ""
        
        print(f"\n📖 Generating comprehensive answer for: '{question}'")
        
        # Get multiple relevant chunks
        results = self.vector_store.similarity_search(question, k=6)
        
        if not results:
            return "No relevant information found."
        
        # Combine content from multiple sources
        combined_content = ""
        sources_used = set()
        
        for doc in results:
            source = doc.metadata.get('source', 'unknown')
            sources_used.add(source)
            combined_content += doc.page_content + " "
        
        # Refine the combined response
        if self.response_refiner:
            refined_answer = self.response_refiner.refine_response(question, combined_content, target_length)
            
            print(f"✅ Generated answer using {len(sources_used)} sources: {', '.join(sources_used)}")
            return refined_answer
        else:
            # Basic truncation if no refiner
            words = combined_content.split()
            max_words = {"short": 50, "medium": 100, "long": 150}[target_length]
            if len(words) > max_words:
                combined_content = ' '.join(words[:max_words]) + "..."
            
            return combined_content
    
    def get_system_status(self) -> Dict:
        """Get current system status and statistics."""
        status = {
            'ready': self.is_ready,
            'documents_loaded': self.documents_loaded,
            'refinement_enabled': self.use_refinement,
            'pdf_directory': self.pdf_directory,
            'vector_db_path': self.vector_db_path
        }
        
        if self.is_ready:
            status.update(self.stats)
        
        return status
    
    def interactive_query(self):
        """Interactive query interface for the RAG system."""
        if not self.is_ready:
            print("❌ RAG system not ready. Please run setup() first.")
            return
        
        print("\n" + "="*60)
        print("🤖 INTERACTIVE ETHICAL AI RAG SYSTEM")
        print("="*60)
        print("Ask questions about AI ethics. Type 'quit' to exit.")
        print("Available commands:")
        print("  - 'compare [topic]' - Compare sources on a topic")
        print("  - 'keywords [terms]' - Search for specific keywords")
        print("  - 'comprehensive [question]' - Get detailed answer")
        print("  - 'status' - Show system status")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n🔍 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'status':
                    status = self.get_system_status()
                    print(f"\n📊 System Status:")
                    for key, value in status.items():
                        print(f"   {key}: {value}")
                    continue
                
                if user_input.lower().startswith('compare '):
                    topic = user_input[8:].strip()
                    sources = self.compare_sources(topic)
                    
                    for source, docs in sources.items():
                        print(f"\n📄 {source.upper()}:")
                        for i, doc in enumerate(docs[:2]):  # Show max 2 per source
                            preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                            print(f"   {i+1}. {preview}")
                    continue
                
                if user_input.lower().startswith('keywords '):
                    keywords = user_input[9:].strip()
                    results = self.search_keywords(keywords)
                    
                    for i, (doc, score) in enumerate(results):
                        source = doc.metadata.get('source', 'unknown')
                        preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                        print(f"\n{i+1}. [{source}] (Score: {score:.3f})")
                        print(f"   {preview}")
                    continue
                
                if user_input.lower().startswith('comprehensive '):
                    question = user_input[13:].strip()
                    answer = self.get_comprehensive_answer(question, "medium")
                    print(f"\n💡 Comprehensive Answer:\n{answer}")
                    continue
                
                # Default: simple question answering
                if user_input:
                    results = self.ask_ethics_question(user_input, num_results=3)
                    
                    for i, doc in enumerate(results):
                        source = doc.metadata.get('source', 'unknown')
                        print(f"\n{i+1}. From {source}:")
                        print(f"   {doc.page_content}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Example usage
    print("Initializing Ethical AI RAG System...")
    
    # Create RAG system
    rag = EthicalAIRAG(
        pdf_directory=".",  # Current directory
        use_refinement=True
    )
    
    # Setup the system
    success = rag.setup()
    
    if success:
        # Start interactive session
        rag.interactive_query()
    else:
        print("Failed to set up RAG system. Please check your PDF files and try again.") 