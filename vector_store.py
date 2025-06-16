import os
import warnings
from typing import List, Optional
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set environment variables for stable operation
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TRANSFORMERS_CACHE', './cache')


class VectorStore:
    """Manage vector embeddings and similarity search using Chroma."""
    
    def __init__(self, persist_directory: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.embeddings = None
        self.vectorstore = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model with robust error handling."""
        print("\n" + "="*50)
        print("🧠 CREATING VECTOR STORE")
        print("="*50)
        print("Loading embeddings model (this may take a moment)...")
        
        # List of models to try in order of preference
        model_options = [
            self.model_name,
            "all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L6-v2",
            "all-mpnet-base-v2"
        ]
        
        for model_name in model_options:
            try:
                print(f"Attempting to load model: {model_name}")
                
                # Configure model kwargs for stability
                model_kwargs = {
                    'device': 'cpu',  # Force CPU to avoid GPU issues
                    'trust_remote_code': False,  # Security
                }
                
                # Configure encode kwargs for stability
                encode_kwargs = {
                    'normalize_embeddings': True,
                    'batch_size': 32,  # Smaller batch size for memory efficiency
                    'show_progress_bar': False,  # Disable for cleaner output
                    'convert_to_tensor': True,
                    'device': 'cpu'  # Ensure CPU usage
                }
                
                # Try to initialize with error handling
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    cache_folder='./cache'  # Use local cache
                )
                
                # Test the model with a simple embedding
                test_text = "This is a test sentence."
                test_embedding = self.embeddings.embed_query(test_text)
                
                if test_embedding and len(test_embedding) > 0:
                    print(f"✅ Embeddings model '{model_name}' loaded successfully")
                    print(f"   Embedding dimension: {len(test_embedding)}")
                    self.model_name = model_name  # Update to working model
                    return
                else:
                    print(f"⚠️ Model '{model_name}' loaded but test embedding failed")
                    continue
                    
            except Exception as e:
                print(f"❌ Failed to load model '{model_name}': {e}")
                if "meta tensor" in str(e).lower():
                    print("   This is a PyTorch meta tensor issue - trying next model...")
                elif "memory" in str(e).lower():
                    print("   Memory issue detected - trying lighter model...")
                elif "connection" in str(e).lower():
                    print("   Network issue - trying cached model...")
                continue
        
        # If all models failed, raise an error
        raise RuntimeError(
            "Failed to load any embedding model. This could be due to:\n"
            "1. PyTorch version incompatibility\n"
            "2. Insufficient memory\n"
            "3. Network connectivity issues\n"
            "4. Missing dependencies\n\n"
            "Try: pip install --upgrade torch sentence-transformers"
        )
    
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
                
                # Verify the vectorstore is working
                collection_count = self.vectorstore._collection.count()
                print(f"✅ Loaded existing vectorstore with {collection_count} documents")
                return True
                
            except Exception as e:
                print(f"⚠️ Failed to load existing vectorstore: {e}")
                print("Creating new vectorstore...")
        
        # Create new vectorstore
        try:
            print(f"🔄 Creating vectorstore from {len(documents)} documents...")
            
            # Process documents in smaller batches to avoid memory issues
            batch_size = 50  # Smaller batches for stability
            
            if len(documents) <= batch_size:
                # Small number of documents - process all at once
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                # Large number of documents - process in batches
                print(f"Processing {len(documents)} documents in batches of {batch_size}...")
                
                # Create initial vectorstore with first batch
                first_batch = documents[:batch_size]
                self.vectorstore = Chroma.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
                
                # Add remaining documents in batches
                for i in range(batch_size, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    print(f"   Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                    
                    try:
                        self.vectorstore.add_documents(batch)
                    except Exception as e:
                        print(f"   ⚠️ Warning: Failed to add batch {i//batch_size + 1}: {e}")
                        continue
            
            # Persist the vectorstore
            try:
                self.vectorstore.persist()
                print(f"✅ Vectorstore created and persisted to {self.persist_directory}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to persist vectorstore: {e}")
                print("   Vectorstore created but may not persist between sessions")
            
            # Verify creation
            try:
                final_count = self.vectorstore._collection.count()
                print(f"📊 Total vectors: {final_count}")
                return True
            except Exception as e:
                print(f"⚠️ Warning: Could not verify vector count: {e}")
                return True  # Assume success if we got this far
            
        except Exception as e:
            print(f"❌ Failed to create vectorstore: {e}")
            
            # Provide specific troubleshooting based on error type
            error_str = str(e).lower()
            if "memory" in error_str or "out of memory" in error_str:
                print("\n💡 Memory issue detected. Try:")
                print("   - Reducing the number of documents")
                print("   - Using a smaller embedding model")
                print("   - Restarting the application")
            elif "meta tensor" in error_str:
                print("\n💡 PyTorch tensor issue detected. Try:")
                print("   - Updating PyTorch: pip install --upgrade torch")
                print("   - Updating sentence-transformers: pip install --upgrade sentence-transformers")
                print("   - Restarting the application")
            elif "connection" in error_str or "network" in error_str:
                print("\n💡 Network issue detected. Try:")
                print("   - Checking internet connection")
                print("   - Using a different embedding model")
                print("   - Trying again later")
            
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


# Utility function for testing
def test_embeddings_model(model_name: str = "all-MiniLM-L6-v2") -> bool:
    """Test if an embeddings model can be loaded successfully."""
    try:
        print(f"Testing embeddings model: {model_name}")
        
        model_kwargs = {'device': 'cpu', 'trust_remote_code': False}
        encode_kwargs = {'normalize_embeddings': True, 'device': 'cpu'}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Test embedding
        test_embedding = embeddings.embed_query("Test sentence")
        
        if test_embedding and len(test_embedding) > 0:
            print(f"✅ Model {model_name} works correctly")
            return True
        else:
            print(f"❌ Model {model_name} failed test")
            return False
            
    except Exception as e:
        print(f"❌ Model {model_name} failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage with error handling
    from pdf_extractor import PDFExtractor
    from document_processor import DocumentProcessor
    
    print("Starting vector store creation with enhanced error handling...")
    
    # Test embeddings models first
    print("\n🧪 Testing available embedding models...")
    models_to_test = [
        "all-MiniLM-L6-v2",
        "paraphrase-MiniLM-L6-v2",
        "all-mpnet-base-v2"
    ]
    
    working_models = []
    for model in models_to_test:
        if test_embeddings_model(model):
            working_models.append(model)
    
    if not working_models:
        print("❌ No working embedding models found!")
        print("Try: pip install --upgrade torch sentence-transformers")
        exit(1)
    
    print(f"✅ Found {len(working_models)} working models: {working_models}")
    
    # Extract and process documents
    extractor = PDFExtractor()
    documents_text = extractor.extract_from_directory()
    
    if documents_text:
        processor = DocumentProcessor()
        rag_documents = processor.create_rag_documents(documents_text)
        
        if rag_documents:
            # Create vector store with best available model
            vectorstore = VectorStore(model_name=working_models[0])
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
                print("❌ Vector store creation failed")
        else:
            print("❌ No documents processed")
    else:
        print("❌ No documents extracted") 
