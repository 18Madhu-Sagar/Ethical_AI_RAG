"""
Simplified but Reliable Vector Store
Works with Python 3.13 and current dependencies
"""

import json
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SimpleDocument:
    """Simple document class compatible with our system."""
    def __init__(self, page_content: str, metadata: Optional[Dict] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class AdvancedVectorStore:
    """Simplified but reliable vector store using TF-IDF."""
    
    def __init__(self, persist_directory: str = "chroma_advanced_db", model_name: str = "tfidf", device: str = "cpu"):
        """Initialize the vector store."""
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        self.model_name = model_name
        self.device = device
        
        # TF-IDF vectorizer (reliable and fast)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # Storage
        self.documents = []
        self.vectors = None
        self.metadata = []
        self.is_ready = False
        
        # Try to load existing data
        self._load_existing()
        logger.info(f"✅ Initialized vector store: {model_name}")
    
    def _load_existing(self):
        """Load existing vectorstore if it exists."""
        try:
            data_file = self.persist_directory / "data.json"
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.documents = data.get('documents', [])
                self.metadata = data.get('metadata', [])
                self.is_ready = data.get('is_ready', False)
                
                # Load vectorizer and vectors if they exist
                vectorizer_file = self.persist_directory / "vectorizer.pkl"
                vectors_file = self.persist_directory / "vectors.pkl"
                
                if self.is_ready and vectorizer_file.exists() and vectors_file.exists():
                    with open(vectorizer_file, 'rb') as f:
                        self.vectorizer = pickle.load(f)
                    with open(vectors_file, 'rb') as f:
                        self.vectors = pickle.load(f)
                
                if self.documents:
                    logger.info(f"✅ Loaded existing vectorstore with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"❌ Error loading existing vectorstore: {e}")
    
    def create_vectorstore(self, documents: List, force_recreate: bool = False) -> bool:
        """Create vectorstore from documents."""
        try:
            if force_recreate:
                self.clear()
            
            # Extract text content from documents
            texts = []
            metadatas = []
            
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    texts.append(doc.page_content)
                    metadatas.append(getattr(doc, 'metadata', {}))
                elif isinstance(doc, str):
                    texts.append(doc)
                    metadatas.append({})
                else:
                    texts.append(str(doc))
                    metadatas.append({})
            
            return self.add_documents(texts, metadatas)
            
        except Exception as e:
            logger.error(f"❌ Error creating vectorstore: {e}")
            return False
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> bool:
        """Add documents to the vector store."""
        try:
            if not texts:
                logger.warning("No texts provided")
                return False
            
            # Clean texts
            clean_texts = [text.strip() for text in texts if text and text.strip()]
            if not clean_texts:
                logger.warning("No valid texts after cleaning")
                return False
            
            # Add to documents
            start_idx = len(self.documents)
            self.documents.extend(clean_texts)
            
            # Add metadata
            if metadatas:
                self.metadata.extend(metadatas)
            else:
                for i, text in enumerate(clean_texts):
                    self.metadata.append({
                        'doc_id': start_idx + i,
                        'length': len(text),
                        'source': 'uploaded_document'
                    })
            
            # Refit vectorizer with all documents
            self.vectors = self.vectorizer.fit_transform(self.documents)
            self.is_ready = True
            
            logger.info(f"✅ Added {len(clean_texts)} documents. Total: {len(self.documents)}")
            
            # Persist the data
            self._persist()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 3) -> List:
        """Search for similar documents."""
        try:
            if not self.is_ready or not self.documents:
                logger.warning("Vector store not ready or no documents")
                return []
            
            if not query.strip():
                logger.warning("Empty query")
                return []
            
            # Transform query
            query_vector = self.vectorizer.transform([query.strip()])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    doc_text = self.documents[idx]
                    metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                    doc = SimpleDocument(doc_text, metadata)
                    results.append(doc)
            
            logger.info(f"🔍 Found {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 3) -> List[Tuple]:
        """Search with scores."""
        try:
            if not self.is_ready or not self.documents:
                return []
            
            query_vector = self.vectorizer.transform([query.strip()])
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    doc_text = self.documents[idx]
                    score = float(similarities[idx])
                    metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                    doc = SimpleDocument(doc_text, metadata)
                    results.append((doc, score))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in similarity search with score: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        return {
            'status': 'ready' if self.is_ready else 'not_ready',
            'num_documents': len(self.documents),
            'embedding_dimension': self.vectors.shape[1] if self.vectors is not None else 0,
            'model_name': self.model_name
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            'num_documents': len(self.documents),
            'embedding_dimension': self.vectors.shape[1] if self.vectors is not None else 0,
            'is_ready': self.is_ready,
            'model_name': self.model_name,
            'persist_directory': str(self.persist_directory)
        }
    
    def clear(self):
        """Clear all data from the vector store."""
        try:
            self.documents = []
            self.vectors = None
            self.metadata = []
            self.is_ready = False
            
            # Clear persisted files
            data_file = self.persist_directory / "data.json"
            vectorizer_file = self.persist_directory / "vectorizer.pkl"
            vectors_file = self.persist_directory / "vectors.pkl"
            
            for file_path in [data_file, vectorizer_file, vectors_file]:
                if file_path.exists():
                    file_path.unlink()
            
            logger.info("✅ Vector store cleared")
            
        except Exception as e:
            logger.error(f"❌ Error clearing vector store: {e}")
    
    def _persist(self):
        """Persist data to disk."""
        try:
            # Save documents and metadata
            data = {
                'documents': self.documents,
                'metadata': self.metadata,
                'is_ready': self.is_ready
            }
            
            data_file = self.persist_directory / "data.json"
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Save vectorizer and vectors
            if self.is_ready and self.vectors is not None:
                vectorizer_file = self.persist_directory / "vectorizer.pkl"
                vectors_file = self.persist_directory / "vectors.pkl"
                
                with open(vectorizer_file, 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                with open(vectors_file, 'wb') as f:
                    pickle.dump(self.vectors, f)
            
            logger.info("✅ Vector store data persisted")
            
        except Exception as e:
            logger.error(f"❌ Error persisting vector store: {e}") 