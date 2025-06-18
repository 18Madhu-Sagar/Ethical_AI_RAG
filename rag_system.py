"""
Enhanced RAG System with Simple LLM
Works with Python 3.13 and includes simple text generation
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from pdf_extractor import PDFExtractor
from document_processor import DocumentProcessor
from vector_store import AdvancedVectorStore

logger = logging.getLogger(__name__)

class AdvancedRAGSystem:
    """Enhanced RAG system with simple text generation."""
    
    def __init__(self,
                 pdf_directory: str = ".",
                 vector_db_path: str = "./simple_vector_db",
                 embedding_model: str = "tfidf",
                 llm_provider: str = "enhanced_simple",
                 llm_model: str = "simple",
                 use_streaming: bool = False):
        """Initialize the RAG system."""
        
        self.pdf_directory = pdf_directory
        self.vector_db_path = vector_db_path
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.use_streaming = use_streaming
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.document_processor = DocumentProcessor()
        self.vector_store = AdvancedVectorStore(
            persist_directory=vector_db_path,
            model_name=embedding_model
        )
        
        # Initialize LLM
        self.llm_pipeline = None
        self.tokenizer = None
        self._initialize_llm()
        
        # System state
        self.is_ready = False
        self.documents_loaded = False
        self.stats = {}
        
        logger.info("✅ Enhanced RAG system initialized")
    
    def _initialize_llm(self):
        """Initialize the LLM with simple generation."""
        # For deployment, always use enhanced simple generation to avoid atexit issues
        self.llm_provider = "enhanced_simple"
        logger.info("📝 Using enhanced simple text generation for deployment")
        self.llm_pipeline = None
    
    def setup(self, force_rebuild: bool = False) -> bool:
        """Set up the RAG system."""
        print("\n" + "="*60)
        print("🚀 SETTING UP SIMPLIFIED RAG SYSTEM")
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
                'vector_info': self.vector_store.get_collection_info(),
                'llm_provider': self.llm_provider,
                'llm_model': self.llm_model
            }
            
            print("\n" + "="*60)
            print("✅ SIMPLIFIED RAG SYSTEM SETUP COMPLETE")
            print("="*60)
            self._print_setup_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Setup failed with error: {e}")
            logger.error(f"Setup error: {e}")
            return False
    
    def _print_setup_summary(self):
        """Print a summary of the setup process."""
        print(f"📊 SETUP SUMMARY:")
        print(f"   📄 PDF files processed: {self.stats['pdf_files']}")
        print(f"   📋 Document chunks created: {self.stats['total_chunks']}")
        print(f"   🧠 Vector store status: {self.stats['vector_info']['status']}")
        print(f"   📏 Average chunk length: {self.stats['chunk_stats']['avg_chunk_length']:.1f} characters")
        print(f"   🤖 LLM provider: {self.stats['llm_provider']}")
        print(f"   🎯 Embedding model: {self.embedding_model}")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an LLM-generated answer."""
        if not self.is_ready:
            return {
                "error": "RAG system not ready. Please process documents first.",
                "answer": "",
                "sources": [],
                "confidence": 0.0
            }
        
        try:
            print(f"\n🔍 Processing question: '{question}'")
            
            # Get relevant documents
            results = self.vector_store.similarity_search(question, k=5)
            
            if not results:
                return {
                    "answer": "I couldn't find any relevant information in the uploaded documents.",
                    "sources": [],
                    "confidence": 0.0,
                    "num_sources": 0
                }
            
            # Prepare context from retrieved documents
            context_parts = []
            sources = []
            
            for i, doc in enumerate(results[:3], 1):
                content = doc.page_content
                source_info = {
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
                context_parts.append(content[:500])  # Limit context length
            
            context = "\n\n".join(context_parts)
            
            # Generate answer using LLM
            if self.llm_provider == "enhanced_simple":
                # Enhanced simple generation with better formatting
                answer = self._generate_enhanced_answer(question, context_parts)
                confidence = 0.8
            else:
                # Basic fallback
                answer = self._generate_simple_answer(question, context_parts)
                confidence = 0.7
            
            print(f"✅ Generated answer using {len(sources)} sources with {self.llm_provider}")
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "num_sources": len(sources),
                "llm_model": self.llm_model
            }
            
        except Exception as e:
            logger.error(f"Error in ask_question: {e}")
            return {
                "error": f"Failed to generate answer: {str(e)}",
                "answer": "",
                "sources": [],
                "confidence": 0.0
            }
    
    def _generate_enhanced_answer(self, question: str, context_parts: List[str]) -> str:
        """Generate enhanced answer with better analysis."""
        # Analyze question type
        question_lower = question.lower()
        is_what_question = any(word in question_lower for word in ['what', 'define', 'definition'])
        is_how_question = any(word in question_lower for word in ['how', 'method', 'approach'])
        is_why_question = any(word in question_lower for word in ['why', 'reason', 'because'])
        is_list_question = any(word in question_lower for word in ['list', 'types', 'kinds', 'examples'])
        
        # Extract key concepts from question
        key_terms = []
        ethics_terms = ['ethics', 'ethical', 'bias', 'fairness', 'transparency', 'accountability', 'privacy', 'ai', 'artificial intelligence']
        for term in ethics_terms:
            if term in question_lower:
                key_terms.append(term)
        
        # Generate structured answer
        answer = f"**AI Ethics Expert Analysis:**\n\n"
        
        if is_what_question:
            answer += f"**Definition and Overview:**\n"
        elif is_how_question:
            answer += f"**Methods and Approaches:**\n"
        elif is_why_question:
            answer += f"**Reasoning and Importance:**\n"
        elif is_list_question:
            answer += f"**Key Points and Examples:**\n"
        else:
            answer += f"**Analysis:**\n"
        
        # Combine and analyze context
        combined_context = " ".join(context_parts)
        
        # Extract relevant sentences
        sentences = combined_context.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Filter out very short fragments
                # Check if sentence contains key terms or question-related words
                sentence_lower = sentence.lower()
                relevance_score = 0
                
                for term in key_terms:
                    if term in sentence_lower:
                        relevance_score += 2
                
                # Boost score for question-type words
                if is_what_question and any(word in sentence_lower for word in ['is', 'are', 'means', 'refers']):
                    relevance_score += 1
                elif is_how_question and any(word in sentence_lower for word in ['by', 'through', 'using', 'method']):
                    relevance_score += 1
                elif is_why_question and any(word in sentence_lower for word in ['because', 'since', 'due to', 'important']):
                    relevance_score += 1
                
                if relevance_score > 0:
                    relevant_sentences.append((sentence, relevance_score))
        
        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in relevant_sentences[:5]]
        
        if top_sentences:
            answer += "\n".join([f"• {sentence.strip()}." for sentence in top_sentences])
        else:
            # Fallback to first parts of context
            for i, content in enumerate(context_parts[:2], 1):
                summary = content[:200] + "..." if len(content) > 200 else content
                answer += f"\n• **From Source {i}:** {summary}"
        
        answer += f"\n\n**Key Terms Identified:** {', '.join(key_terms) if key_terms else 'AI Ethics concepts'}"
        answer += f"\n\n**Based on {len(context_parts)} relevant document sections**"
        
        return answer
    
    def _generate_simple_answer(self, question: str, context_parts: List[str]) -> str:
        """Generate simple answer by combining context."""
        answer_parts = []
        for i, content in enumerate(context_parts, 1):
            if len(content) > 300:
                summary = content[:300] + "..."
            else:
                summary = content
            answer_parts.append(f"**Source {i}:** {summary}")
        
        answer = f"**Answer based on your documents:**\n\n"
        answer += "\n\n".join(answer_parts)
        return answer
    
    def _create_qa_chain(self):
        """Create QA chain (simplified version)."""
        # This is a simplified version that doesn't require complex LLM setup
        self.is_ready = True
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'ready': self.is_ready,
            'documents_loaded': self.documents_loaded,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'embedding_model': self.embedding_model,
            'vector_store_stats': self.vector_store.get_stats(),
            'pdf_directory': self.pdf_directory,
            'vector_db_path': self.vector_db_path
        }

# Main RAG system class
EthicalAIRAG = AdvancedRAGSystem 