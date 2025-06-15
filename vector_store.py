import os
import warnings
from typing import List, Optional
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)


class VectorStore:
    """Manage vector embeddings and similarity search using Chroma."""
    
    def __init__(self, persist_directory: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.embeddings = None
        self.vectorstore = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        print("\n" + "="*50)
        print("🧠 CREATING VECTOR STORE")
        print("="*50)
        print("Loading embeddings model (this may take a moment)...")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"✅ Embeddings model '{self.model_name}' loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load embeddings model: {e}")
            raise
    
    def create_vectorstore(self, documents: List[Document], force_recreate: bool = False) -> bool:
        """Create or load vector store from documents."""
        if not documents:
            print("❌ No documents provided for vectorstore creation")
            return False
        
        # Check if vectorstore already exists
        if os.path.exists(self.persist_directory) and not force_recreate:
            print(f"📂 Loading existing vectorstore from {self.persist_directory}")
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                print(f"✅ Loaded existing vectorstore with {self.vectorstore._collection.count()} documents")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load existing vectorstore: {e}")
                print("Creating new vectorstore...")
        
        # Create new vectorstore
        try:
            print(f"🔄 Creating vectorstore from {len(documents)} documents...")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Persist the vectorstore
            self.vectorstore.persist()
            print(f"✅ Vectorstore created and persisted to {self.persist_directory}")
            print(f"📊 Total vectors: {len(documents)}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create vectorstore: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search in the vector store."""
        if not self.vectorstore:
            print("❌ Vectorstore not initialized")
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"❌ Similarity search failed: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Perform similarity search with relevance scores."""
        if not self.vectorstore:
            print("❌ Vectorstore not initialized")
            return []
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"❌ Similarity search with score failed: {e}")
            return []
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add new documents to existing vectorstore."""
        if not self.vectorstore:
            print("❌ Vectorstore not initialized")
            return False
        
        try:
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            print(f"✅ Added {len(documents)} new documents to vectorstore")
            return True
        except Exception as e:
            print(f"❌ Failed to add documents: {e}")
            return False
    
    def delete_collection(self) -> bool:
        """Delete the entire vector collection."""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                print("✅ Vector collection deleted")
            
            # Remove persist directory
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                print(f"✅ Persist directory {self.persist_directory} removed")
            
            self.vectorstore = None
            return True
        except Exception as e:
            print(f"❌ Failed to delete collection: {e}")
            return False
    
    def get_collection_info(self) -> dict:
        """Get information about the vector collection."""
        if not self.vectorstore:
            return {"status": "not_initialized"}
        
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "status": "ready",
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.model_name
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    # Example usage
    from pdf_extractor import PDFExtractor
    from document_processor import DocumentProcessor
    
    print("Starting vector store creation...")
    
    # Extract and process documents
    extractor = PDFExtractor()
    documents_text = extractor.extract_from_directory()
    
    if documents_text:
        processor = DocumentProcessor()
        rag_documents = processor.create_rag_documents(documents_text)
        
        if rag_documents:
            # Create vector store
            vectorstore = VectorStore()
            success = vectorstore.create_vectorstore(rag_documents)
            
            if success:
                # Test similarity search
                print("\n" + "="*50)
                print("🔍 TESTING SIMILARITY SEARCH")
                print("="*50)
                
                test_query = "What are the ethical principles of AI?"
                results = vectorstore.similarity_search(test_query, k=3)
                
                print(f"Query: {test_query}")
                print(f"Found {len(results)} relevant chunks:")
                
                for i, doc in enumerate(results):
                    source = doc.metadata.get('source', 'unknown')
                    preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    print(f"\n{i+1}. From {source}:")
                    print(f"   {preview}")
                
                # Show collection info
                info = vectorstore.get_collection_info()
                print(f"\n📊 Collection Info:")
                print(f"   Status: {info['status']}")
                print(f"   Documents: {info.get('document_count', 'unknown')}")
                print(f"   Model: {info.get('embedding_model', 'unknown')}")
    else:
        print("No documents to vectorize.") 