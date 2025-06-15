#!/usr/bin/env python3
"""
Ethical AI RAG System - Main Entry Point

A comprehensive Retrieval-Augmented Generation system for AI ethics documents.
This system extracts text from PDFs, creates vector embeddings, and provides
intelligent query capabilities with response refinement.

Usage:
    python main.py                    # Interactive mode
    python main.py --setup            # Setup only
    python main.py --query "question" # Single query
    python main.py --demo             # Run demo queries
"""

import argparse
import sys
import os
from typing import List, Dict

from rag_system import EthicalAIRAG


def run_demo_queries(rag_system: EthicalAIRAG) -> None:
    """Run a set of demo queries to showcase the system."""
    print("\n" + "="*60)
    print("🎯 RUNNING DEMO QUERIES")
    print("="*60)
    
    demo_questions = [
        "What are the main ethical principles of AI?",
        "How should AI systems ensure fairness?",
        "What is algorithmic bias and how can it be prevented?",
        "What are the privacy concerns with AI systems?",
        "How should AI be governed and regulated?"
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n📝 Demo Query {i}: {question}")
        print("-" * 50)
        
        # Get comprehensive answer
        answer = rag_system.get_comprehensive_answer(question, "medium")
        print(f"💡 Answer: {answer}")
        
        # Show sources
        results = rag_system.ask_ethics_question(question, num_results=2, refine_response=False)
        sources = set(doc.metadata.get('source', 'unknown') for doc in results)
        print(f"📚 Sources: {', '.join(sources)}")
        
        print()  # Add spacing


def single_query_mode(rag_system: EthicalAIRAG, question: str) -> None:
    """Handle a single query and display results."""
    print(f"\n🔍 Query: {question}")
    print("="*60)
    
    # Get comprehensive answer
    answer = rag_system.get_comprehensive_answer(question, "medium")
    print(f"\n💡 Answer:\n{answer}")
    
    # Get detailed results
    results = rag_system.ask_ethics_question(question, num_results=3, refine_response=True)
    
    if results:
        print(f"\n📋 Detailed Results ({len(results)} chunks):")
        print("-" * 40)
        
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'unknown')
            chunk_id = doc.metadata.get('chunk_id', 'unknown')
            print(f"\n{i}. Source: {source} (Chunk {chunk_id})")
            print(f"   Content: {doc.page_content}")
    
    # Show keyword search
    keyword_results = rag_system.search_keywords(question, num_results=3)
    if keyword_results:
        print(f"\n🔎 Related Content (with scores):")
        print("-" * 40)
        for i, (doc, score) in enumerate(keyword_results, 1):
            source = doc.metadata.get('source', 'unknown')
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"{i}. [{source}] Score: {score:.3f}")
            print(f"   {preview}")


def print_system_info(rag_system: EthicalAIRAG) -> None:
    """Print detailed system information and statistics."""
    status = rag_system.get_system_status()
    
    print("\n" + "="*60)
    print("📊 SYSTEM INFORMATION")
    print("="*60)
    
    print(f"System Status: {'✅ Ready' if status['ready'] else '❌ Not Ready'}")
    print(f"PDF Directory: {status['pdf_directory']}")
    print(f"Vector DB Path: {status['vector_db_path']}")
    print(f"Refinement: {'✅ Enabled' if status['refinement_enabled'] else '❌ Disabled'}")
    
    if status['ready']:
        print(f"\n📄 Document Statistics:")
        print(f"   PDF Files Processed: {status['pdf_files']}")
        print(f"   Total Chunks Created: {status['total_chunks']}")
        
        chunk_stats = status['chunk_stats']
        print(f"   Average Chunk Length: {chunk_stats['avg_chunk_length']:.1f} characters")
        print(f"   Min/Max Chunk Length: {chunk_stats['min_chunk_length']}/{chunk_stats['max_chunk_length']}")
        
        print(f"\n📚 Sources: {', '.join(chunk_stats['sources'])}")
        for source, count in chunk_stats['chunks_per_source'].items():
            print(f"   📄 {source}: {count} chunks")
        
        vector_info = status['vector_info']
        print(f"\n🧠 Vector Store:")
        print(f"   Status: {vector_info['status']}")
        print(f"   Document Count: {vector_info.get('document_count', 'unknown')}")
        print(f"   Embedding Model: {vector_info.get('embedding_model', 'unknown')}")


def check_pdf_files(directory: str) -> bool:
    """Check if there are PDF files in the specified directory."""
    if not os.path.exists(directory):
        print(f"❌ Directory '{directory}' does not exist.")
        return False
    
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ No PDF files found in '{directory}'.")
        print("Please add some PDF files containing AI ethics content to get started.")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")
    return True


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Ethical AI RAG System - Query AI ethics documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                              # Interactive mode
    python main.py --setup                      # Setup system only
    python main.py --query "What is AI bias?"   # Single query
    python main.py --demo                       # Run demo queries
    python main.py --info                       # Show system info
    python main.py --dir /path/to/pdfs          # Use different PDF directory
        """
    )
    
    parser.add_argument(
        '--dir', 
        default='.',
        help='Directory containing PDF files (default: current directory)'
    )
    
    parser.add_argument(
        '--setup', 
        action='store_true',
        help='Setup the system (extract PDFs, create vectors) and exit'
    )
    
    parser.add_argument(
        '--query', 
        type=str,
        help='Ask a single question and exit'
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Run demo queries showcasing the system'
    )
    
    parser.add_argument(
        '--info', 
        action='store_true',
        help='Show detailed system information'
    )
    
    parser.add_argument(
        '--force-rebuild', 
        action='store_true',
        help='Force rebuild of vector database'
    )
    
    parser.add_argument(
        '--no-refinement', 
        action='store_true',
        help='Disable response refinement'
    )
    
    args = parser.parse_args()
    
    # Check for PDF files
    if not check_pdf_files(args.dir):
        sys.exit(1)
    
    print("🚀 Initializing Ethical AI RAG System...")
    print(f"📁 PDF Directory: {args.dir}")
    
    # Initialize RAG system
    rag = EthicalAIRAG(
        pdf_directory=args.dir,
        use_refinement=not args.no_refinement
    )
    
    # Setup the system
    print("\n⚙️ Setting up system...")
    success = rag.setup(force_rebuild=args.force_rebuild)
    
    if not success:
        print("❌ Failed to set up RAG system. Please check your PDF files and try again.")
        sys.exit(1)
    
    # Handle different modes
    if args.setup:
        print("✅ Setup complete! You can now run queries.")
        return
    
    if args.info:
        print_system_info(rag)
        return
    
    if args.demo:
        run_demo_queries(rag)
        return
    
    if args.query:
        single_query_mode(rag, args.query)
        return
    
    # Default: interactive mode
    print("\n🎯 Entering interactive mode...")
    print("Type 'help' for available commands or 'quit' to exit.")
    
    try:
        rag.interactive_query()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main() 