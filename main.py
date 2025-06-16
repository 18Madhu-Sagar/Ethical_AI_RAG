#!/usr/bin/env python3
"""
Ethical AI RAG System - Command Line Interface
Enhanced version with robust error handling for deployment environments.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add current directory to Python path for reliable imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def safe_import():
    """Safely import RAG system components with detailed error reporting."""
    try:
        from rag_system import EthicalAIRAG
        from pdf_extractor import PDFExtractor
        from document_processor import DocumentProcessor
        from vector_store import VectorStore
        from response_refiner import ResponseRefiner
        return True, None
    except ImportError as e:
        error_msg = f"""
❌ IMPORT ERROR: {e}

TROUBLESHOOTING:
1. Ensure all required files are present:
   - rag_system.py
   - pdf_extractor.py
   - document_processor.py
   - vector_store.py
   - response_refiner.py

2. Install dependencies:
   pip install -r requirements.txt

3. Check Python path and current directory
        """
        return False, error_msg
    except Exception as e:
        return False, f"Unexpected import error: {e}"

def setup_environment():
    """Set up environment variables for stable operation."""
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('TRANSFORMERS_CACHE', './cache')
    
    # Create cache directory if it doesn't exist
    cache_dir = Path('./cache')
    cache_dir.mkdir(exist_ok=True)

def setup_system(pdf_directory: str = ".", force_rebuild: bool = False, use_refinement: bool = True) -> bool:
    """Set up the RAG system with enhanced error handling."""
    imports_ok, error_msg = safe_import()
    if not imports_ok:
        print(error_msg)
        return False
    
    # Import after successful check
    from rag_system import EthicalAIRAG
    
    try:
        print("\n" + "="*60)
        print("🚀 ETHICAL AI RAG SYSTEM SETUP")
        print("="*60)
        
        # Validate PDF directory
        pdf_path = Path(pdf_directory)
        if not pdf_path.exists():
            print(f"❌ Directory not found: {pdf_directory}")
            return False
        
        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_directory}")
            print("Please add PDF files to the directory and try again.")
            return False
        
        print(f"📄 Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            print(f"   - {pdf_file.name}")
        
        # Initialize and setup RAG system
        vector_db_path = pdf_path / "chroma_db"
        rag_system = EthicalAIRAG(
            pdf_directory=str(pdf_path),
            vector_db_path=str(vector_db_path),
            use_refinement=use_refinement
        )
        
        print(f"\n🔧 Processing documents (force_rebuild={force_rebuild})...")
        success = rag_system.setup(force_rebuild=force_rebuild)
        
        if success:
            print("\n✅ Setup completed successfully!")
            print("\nYou can now:")
            print("  1. Use CLI: python main.py --query 'Your question here'")
            print("  2. Start web UI: streamlit run streamlit_app.py")
            print("  3. Run interactive mode: python main.py --interactive")
            return True
        else:
            print("\n❌ Setup failed. Check error messages above.")
            return False
            
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        print("\nTROUBLESHOoting tips:")
        print("- Ensure sufficient memory (at least 2GB available)")
        print("- Check internet connection for model downloads")
        print("- Verify PDF files are readable and not corrupted")
        return False

def query_system(question: str, pdf_directory: str = ".", num_results: int = 3, query_type: str = "simple") -> bool:
    """Query the RAG system with error handling."""
    imports_ok, error_msg = safe_import()
    if not imports_ok:
        print(error_msg)
        return False
    
    from rag_system import EthicalAIRAG
    
    try:
        # Check if vector store exists
        vector_db_path = Path(pdf_directory) / "chroma_db"
        if not vector_db_path.exists():
            print("❌ Vector store not found. Please run setup first:")
            print("   python main.py --setup")
            return False
        
        # Initialize RAG system
        rag_system = EthicalAIRAG(
            pdf_directory=pdf_directory,
            vector_db_path=str(vector_db_path),
            use_refinement=False  # Disable for CLI for speed
        )
        
        # Load existing vector store
        print("🔍 Loading vector store...")
        if not rag_system.vector_store.vectorstore:
            success = rag_system.setup(force_rebuild=False)
            if not success:
                print("❌ Failed to load vector store")
                return False
        
        rag_system.is_ready = True
        
        print(f"\n💭 Question: {question}")
        print("-" * 60)
        
        if query_type == "comprehensive":
            answer = rag_system.get_comprehensive_answer(question)
            if answer:
                print(f"📝 Answer:\n{answer}")
            else:
                print("❌ No comprehensive answer could be generated")
        else:
            # Simple query
            results = rag_system.ask_ethics_question(question, num_results=num_results, refine_response=False)
            
            if results:
                print(f"📚 Found {len(results)} relevant sources:")
                for i, doc in enumerate(results, 1):
                    source = doc.metadata.get('source', 'unknown')
                    chunk_id = doc.metadata.get('chunk_id', 'N/A')
                    print(f"\n📄 Source {i}: {source} (Chunk {chunk_id})")
                    print("-" * 40)
                    
                    # Show first 300 characters
                    content = doc.page_content
                    preview = content[:300] + "..." if len(content) > 300 else content
                    print(preview)
            else:
                print("❌ No relevant information found")
                print("Try rephrasing your question or using different keywords")
        
        return True
        
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False

def interactive_mode(pdf_directory: str = "."):
    """Run interactive query mode with error handling."""
    imports_ok, error_msg = safe_import()
    if not imports_ok:
        print(error_msg)
        return
    
    from rag_system import EthicalAIRAG
    
    try:
        # Initialize system
        rag_system = EthicalAIRAG(
            pdf_directory=pdf_directory,
            vector_db_path=os.path.join(pdf_directory, "chroma_db"),
            use_refinement=False
        )
        
        # Check if setup is needed
        if not rag_system.vector_store.vectorstore:
            print("🔧 Vector store not found. Running setup...")
            success = rag_system.setup(force_rebuild=False)
            if not success:
                print("❌ Setup failed. Please run: python main.py --setup")
                return
        
        rag_system.is_ready = True
        
        print("\n" + "="*60)
        print("🤖 INTERACTIVE AI ETHICS Q&A")
        print("="*60)
        print("Ask questions about AI ethics. Type 'quit' to exit.")
        
        while True:
            try:
                question = input("\n💭 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                # Process query
                results = rag_system.ask_ethics_question(question, num_results=2)
                
                if results:
                    print(f"\n📚 Found relevant information:")
                    for i, doc in enumerate(results, 1):
                        source = doc.metadata.get('source', 'unknown')
                        print(f"\n📄 From {source}:")
                        content = doc.page_content
                        preview = content[:400] + "..." if len(content) > 400 else content
                        print(preview)
                else:
                    print("\n❌ No relevant information found. Try rephrasing your question.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Interactive mode error: {e}")

def get_system_info(pdf_directory: str = "."):
    """Display system information and status."""
    imports_ok, error_msg = safe_import()
    if not imports_ok:
        print(error_msg)
        return
    
    from rag_system import EthicalAIRAG
    
    print("\n" + "="*60)
    print("ℹ️  SYSTEM INFORMATION")
    print("="*60)
    
    # Check PDF files
    pdf_path = Path(pdf_directory)
    pdf_files = list(pdf_path.glob("*.pdf")) if pdf_path.exists() else []
    print(f"📄 PDF Directory: {pdf_path.absolute()}")
    print(f"📄 PDF Files Found: {len(pdf_files)}")
    
    for pdf_file in pdf_files:
        size_mb = pdf_file.stat().st_size / 1024 / 1024
        print(f"   - {pdf_file.name} ({size_mb:.1f} MB)")
    
    # Check vector store
    vector_db_path = pdf_path / "chroma_db"
    print(f"\n🧠 Vector Store: {vector_db_path.absolute()}")
    print(f"🧠 Vector Store Exists: {'✅ Yes' if vector_db_path.exists() else '❌ No'}")
    
    if vector_db_path.exists():
        try:
            rag_system = EthicalAIRAG(
                pdf_directory=str(pdf_path),
                vector_db_path=str(vector_db_path),
                use_refinement=False
            )
            
            status = rag_system.get_system_status()
            print(f"🧠 Status: {status.get('vector_info', {}).get('status', 'unknown')}")
            print(f"📊 Document Chunks: {status.get('total_chunks', 0)}")
            
            if 'chunk_stats' in status:
                chunk_stats = status['chunk_stats']
                print(f"📏 Average Chunk Length: {chunk_stats.get('avg_chunk_length', 0):.0f} characters")
        except Exception as e:
            print(f"⚠️ Could not load vector store details: {e}")
    
    # Environment info
    print(f"\n🐍 Python Version: {sys.version}")
    print(f"📁 Working Directory: {Path.cwd()}")
    print(f"🔧 Current Script: {Path(__file__).absolute()}")

def main():
    """Main command line interface with enhanced error handling."""
    setup_environment()
    
    parser = argparse.ArgumentParser(
        description="Ethical AI RAG System - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup                          # Setup system with PDFs in current directory
  python main.py --setup --dir ./pdfs           # Setup with PDFs in specific directory
  python main.py --query "What are AI ethics?"   # Ask a question
  python main.py --interactive                   # Start interactive mode
  python main.py --info                         # Show system information
        """
    )
    
    parser.add_argument("--setup", action="store_true", 
                      help="Set up the RAG system by processing PDFs")
    parser.add_argument("--query", type=str, 
                      help="Ask a question about AI ethics")
    parser.add_argument("--interactive", action="store_true",
                      help="Start interactive Q&A mode")
    parser.add_argument("--info", action="store_true",
                      help="Show system information and status")
    parser.add_argument("--dir", type=str, default=".",
                      help="Directory containing PDF files (default: current directory)")
    parser.add_argument("--force-rebuild", action="store_true",
                      help="Force rebuild of vector store (use with --setup)")
    parser.add_argument("--no-refinement", action="store_true",
                      help="Disable response refinement for faster processing")
    parser.add_argument("--num-results", type=int, default=3,
                      help="Number of results to return (default: 3)")
    parser.add_argument("--comprehensive", action="store_true",
                      help="Generate comprehensive answer (use with --query)")
    
    args = parser.parse_args()
    
    # Validate directory
    if not Path(args.dir).exists():
        print(f"❌ Directory not found: {args.dir}")
        sys.exit(1)
    
    success = True
    
    try:
        if args.setup:
            success = setup_system(
                pdf_directory=args.dir,
                force_rebuild=args.force_rebuild,
                use_refinement=not args.no_refinement
            )
        elif args.query:
            query_type = "comprehensive" if args.comprehensive else "simple"
            success = query_system(
                question=args.query,
                pdf_directory=args.dir,
                num_results=args.num_results,
                query_type=query_type
            )
        elif args.interactive:
            interactive_mode(pdf_directory=args.dir)
        elif args.info:
            get_system_info(pdf_directory=args.dir)
        else:
            parser.print_help()
            print("\n💡 Quick start: python main.py --setup")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if os.getenv('DEBUG'):
            import traceback
            traceback.print_exc()
        success = False
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 
