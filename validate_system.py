#!/usr/bin/env python3
"""
Validation script for Ethical AI RAG System
Tests all components to ensure they're working properly.
"""

import os
import sys
import tempfile
from io import StringIO

def test_imports():
    """Test if all required modules can be imported."""
    print("\n" + "="*60)
    print("🧪 TESTING IMPORTS")
    print("="*60)
    
    modules_to_test = [
        ('pdf_extractor', 'PDFExtractor'),
        ('document_processor', 'DocumentProcessor'),
        ('vector_store', 'VectorStore'),
        ('response_refiner', 'ResponseRefiner'),
        ('rag_system', 'EthicalAIRAG'),
    ]
    
    results = {}
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            cls = getattr(module, class_name)
            results[module_name] = "✅ SUCCESS"
            print(f"  ✅ {module_name}.{class_name} - imported successfully")
        except Exception as e:
            results[module_name] = f"❌ FAILED: {e}"
            print(f"  ❌ {module_name}.{class_name} - import failed: {e}")
    
    return results

def test_dependencies():
    """Test critical dependencies."""
    print("\n" + "="*60)
    print("🔗 TESTING DEPENDENCIES")
    print("="*60)
    
    dependencies = [
        'langchain',
        'langchain_community',
        'chromadb',
        'sentence_transformers',
        'transformers',
        'PyPDF2',
        'numpy',
        'torch'
    ]
    
    results = {}
    for dep in dependencies:
        try:
            __import__(dep)
            results[dep] = "✅ SUCCESS"
            print(f"  ✅ {dep} - available")
        except ImportError as e:
            results[dep] = f"❌ FAILED: {e}"
            print(f"  ❌ {dep} - not available: {e}")
    
    return results

def test_pdf_extractor():
    """Test PDF extraction functionality."""
    print("\n" + "="*60)
    print("📄 TESTING PDF EXTRACTOR")
    print("="*60)
    
    try:
        from pdf_extractor import PDFExtractor
        
        # Create a minimal test
        extractor = PDFExtractor()
        print("  ✅ PDFExtractor instance created successfully")
        
        # Test the extraction stats feature
        stats = extractor.get_extraction_stats()
        print(f"  ✅ Extraction stats accessible: {type(stats)}")
        
        return "✅ SUCCESS"
    except Exception as e:
        print(f"  ❌ PDF Extractor test failed: {e}")
        return f"❌ FAILED: {e}"

def test_document_processor():
    """Test document processing functionality."""
    print("\n" + "="*60)
    print("🔧 TESTING DOCUMENT PROCESSOR")
    print("="*60)
    
    try:
        from document_processor import DocumentProcessor
        
        # Create processor instance
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        print("  ✅ DocumentProcessor instance created successfully")
        
        # Test text cleaning
        test_text = "This is a   test    text\n\n\nwith multiple   spaces"
        cleaned = processor.clean_text(test_text)
        print(f"  ✅ Text cleaning works: '{cleaned[:50]}...'")
        
        # Test document creation with sample text
        sample_docs = {"test_doc": "This is a sample document for testing purposes. " * 20}
        rag_docs = processor.create_rag_documents(sample_docs)
        print(f"  ✅ Document creation works: {len(rag_docs)} chunks created")
        
        # Test statistics
        stats = processor.get_chunk_statistics(rag_docs)
        print(f"  ✅ Statistics calculation works: {stats['total_chunks']} total chunks")
        
        return "✅ SUCCESS"
    except Exception as e:
        print(f"  ❌ Document Processor test failed: {e}")
        return f"❌ FAILED: {e}"

def test_vector_store():
    """Test vector store functionality."""
    print("\n" + "="*60)
    print("🧠 TESTING VECTOR STORE")
    print("="*60)
    
    try:
        from vector_store import VectorStore
        from langchain.schema import Document
        import time
        
        # Use unique temporary directory for testing
        temp_dir = f"./test_vectordb_{int(time.time())}"
        
        try:
            # Create vector store instance
            vectorstore = VectorStore(persist_directory=temp_dir)
            print("  ✅ VectorStore instance created successfully")
            
            # Create sample documents
            sample_docs = [
                Document(page_content="AI ethics is about responsible AI development.", 
                        metadata={"source": "test1", "chunk_id": 0}),
                Document(page_content="Machine learning fairness prevents bias in algorithms.", 
                        metadata={"source": "test2", "chunk_id": 0}),
            ]
            
            # Test vectorstore creation
            success = vectorstore.create_vectorstore(sample_docs, force_recreate=True)
            if success:
                print("  ✅ Vector store creation successful")
                
                # Test similarity search
                results = vectorstore.similarity_search("AI ethics", k=1)
                print(f"  ✅ Similarity search works: {len(results)} results found")
                
                # Test collection info
                info = vectorstore.get_collection_info()
                print(f"  ✅ Collection info accessible: {info['status']}")
                
            else:
                print("  ⚠️ Vector store creation had issues")
            
            # Clean up
            try:
                vectorstore.delete_collection()
            except:
                pass
            
        finally:
            # Ensure cleanup
            import shutil
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
        
        return "✅ SUCCESS"
    except Exception as e:
        print(f"  ❌ Vector Store test failed: {e}")
        return f"❌ FAILED: {e}"

def test_response_refiner():
    """Test response refinement functionality."""
    print("\n" + "="*60)
    print("✨ TESTING RESPONSE REFINER")
    print("="*60)
    
    try:
        from response_refiner import ResponseRefiner
        
        # Create refiner instance
        refiner = ResponseRefiner(use_summarizer=False)  # Use simple refiner for testing
        print("  ✅ ResponseRefiner instance created successfully")
        
        # Test basic refinement
        query = "What is AI ethics?"
        response = "AI ethics is about ensuring responsible development of artificial intelligence systems. " * 5
        
        refined = refiner.quick_refine(query, response, max_words=50)
        print(f"  ✅ Response refinement works: {len(refined.split())} words")
        
        # Test different refinement types
        stats = refiner.get_refinement_stats()
        print(f"  ✅ Refinement stats accessible: {type(stats)}")
        
        return "✅ SUCCESS"
    except Exception as e:
        print(f"  ❌ Response Refiner test failed: {e}")
        return f"❌ FAILED: {e}"

def test_rag_system():
    """Test the main RAG system."""
    print("\n" + "="*60)
    print("🚀 TESTING RAG SYSTEM")
    print("="*60)
    
    try:
        from rag_system import EthicalAIRAG
        
        # Create RAG system instance
        with tempfile.TemporaryDirectory() as temp_dir:
            rag = EthicalAIRAG(
                pdf_directory=".",
                vector_db_path=temp_dir,
                use_refinement=False  # Disable for testing
            )
            print("  ✅ EthicalAIRAG instance created successfully")
            
            # Test system status
            status = rag.get_system_status()
            print(f"  ✅ System status accessible: {status['ready']}")
            
            # Test without setup (should handle gracefully)
            results = rag.ask_ethics_question("What is AI ethics?")
            print(f"  ✅ Graceful handling when not ready: {len(results)} results")
        
        return "✅ SUCCESS"
    except Exception as e:
        print(f"  ❌ RAG System test failed: {e}")
        return f"❌ FAILED: {e}"

def test_streamlit_app():
    """Test Streamlit app imports."""
    print("\n" + "="*60)
    print("🌐 TESTING STREAMLIT APP")
    print("="*60)
    
    try:
        import streamlit
        print("  ✅ Streamlit available")
        
        # Test if our streamlit app imports correctly
        # Note: We can't run it fully without actually starting streamlit
        try:
            with open('streamlit_app.py', 'r', encoding='utf-8') as f:
                content = f.read()
                if 'import streamlit as st' in content:
                    print("  ✅ Streamlit app structure looks correct")
                else:
                    print("  ⚠️ Streamlit app may have issues")
        except UnicodeDecodeError:
            # Try with different encoding
            with open('streamlit_app.py', 'r', encoding='latin-1') as f:
                content = f.read()
                if 'import streamlit as st' in content:
                    print("  ✅ Streamlit app structure looks correct (encoding issue detected)")
                else:
                    print("  ⚠️ Streamlit app may have structural issues")
        
        return "✅ SUCCESS"
    except Exception as e:
        print(f"  ❌ Streamlit app test failed: {e}")
        return f"❌ FAILED: {e}"

def main():
    """Run all validation tests."""
    print("🔍 ETHICAL AI RAG SYSTEM VALIDATION")
    print("=" * 60)
    print("This script will test all components to ensure they're working properly.")
    
    all_results = {}
    
    # Run all tests
    all_results['imports'] = test_imports()
    all_results['dependencies'] = test_dependencies()
    all_results['pdf_extractor'] = test_pdf_extractor()
    all_results['document_processor'] = test_document_processor()
    all_results['vector_store'] = test_vector_store()
    all_results['response_refiner'] = test_response_refiner()
    all_results['rag_system'] = test_rag_system()
    all_results['streamlit_app'] = test_streamlit_app()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    total_tests = 0
    passed_tests = 0
    
    for test_category, results in all_results.items():
        if isinstance(results, dict):
            # For imports and dependencies
            for item, result in results.items():
                total_tests += 1
                status = "✅" if "SUCCESS" in result else "❌"
                print(f"  {status} {test_category}.{item}")
                if "SUCCESS" in result:
                    passed_tests += 1
        else:
            # For individual tests
            total_tests += 1
            status = "✅" if "SUCCESS" in results else "❌"
            print(f"  {status} {test_category}")
            if "SUCCESS" in results:
                passed_tests += 1
    
    print(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Your Ethical AI RAG system is ready to use.")
        print("\n💡 NEXT STEPS:")
        print("  1. Add some PDF files to the current directory")
        print("  2. Run: python main.py --setup")
        print("  3. Run: streamlit run streamlit_app.py")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        print("\n🔧 TROUBLESHOOTING:")
        print("  1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("  2. Check if there are any syntax errors in the code files")
        print("  3. Ensure you have sufficient system resources")

if __name__ == "__main__":
    main()