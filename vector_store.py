"""
Simplified Vector Store for Emergency Deployment
Uses scikit-learn TF-IDF instead of PyTorch-based embeddings
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle

# Use scikit-learn for embeddings instead of PyTorch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Basic text preprocessing
import re

class SimpleVectorStore:
    """
    Simplified vector store using TF-IDF embeddings instead of transformer models.
    Compatible with Python 3.13 and avoids PyTorch dependency issues.
    """
    
    def __init__(self, persist_directory: str = "./simple_vectordb"):
        """Initialize the simple vector store."""
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # TF-IDF vectorizer for embeddings
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # Optional dimensionality reduction
        self.svd = TruncatedSVD(n_components=300, random_state=42)
        
        # Storage
        self.documents = []
        self.embeddings = None
        self.metadata = []
        self.is_fitted = False
        
        logging.info("SimpleVectorStore initialized successfully")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None) -> bool:
        """Add documents to the vector store."""
        try:
            if not documents:
                return True
            
            # Add to storage
            self.documents.extend(documents)
            
            # Handle metadata
            if metadatas:
                self.metadata.extend(metadatas)
            else:
                self.metadata.extend([{} for _ in documents])
            
            # Refit vectorizer with all documents
            self._refit_vectorizer()
            
            logging.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
            return True
            
        except Exception as e:
            logging.error(f"Error adding documents: {e}")
            return False
    
    def _refit_vectorizer(self):
        """Refit the vectorizer with all documents."""
        try:
            if not self.documents:
                return
            
            # Fit TF-IDF vectorizer
            tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            
            # Apply dimensionality reduction if we have enough features
            if tfidf_matrix.shape[1] > 300:
                self.embeddings = self.svd.fit_transform(tfidf_matrix.toarray())
            else:
                self.embeddings = tfidf_matrix.toarray()
            
            self.is_fitted = True
            logging.info(f"Vectorizer fitted. Embedding shape: {self.embeddings.shape}")
            
        except Exception as e:
            logging.error(f"Error fitting vectorizer: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents."""
        try:
            if not self.is_fitted or self.embeddings is None:
                logging.warning("Vector store not fitted or no embeddings available")
                return []
            
            # Transform query using fitted vectorizer
            query_tfidf = self.vectorizer.transform([query])
            
            # Apply same dimensionality reduction if used
            if hasattr(self.svd, 'components_'):
                query_embedding = self.svd.transform(query_tfidf.toarray())
            else:
                query_embedding = query_tfidf.toarray()
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents) and similarities[idx] > 0.01:
                    results.append((
                        self.documents[idx],
                        float(similarities[idx]),
                        self.metadata[idx] if idx < len(self.metadata) else {}
                    ))
            
            logging.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logging.error(f"Error in similarity search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            'total_documents': len(self.documents),
            'is_fitted': self.is_fitted,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'vectorizer_features': len(self.vectorizer.vocabulary_) if self.is_fitted else 0,
            'persist_directory': str(self.persist_directory)
        }
    
    def clear(self):
        """Clear all data from the vector store."""
        try:
            self.documents = []
            self.embeddings = None
            self.metadata = []
            self.is_fitted = False
            logging.info("Vector store cleared successfully")
        except Exception as e:
            logging.error(f"Error clearing vector store: {e}")

# Compatibility wrapper to match the original interface
class VectorStore(SimpleVectorStore):
    """Compatibility wrapper for the original VectorStore interface."""
    
    def __init__(self, persist_directory: str = "./simple_vectordb"):
        super().__init__(persist_directory)
        self.collection_name = "simple_collection"
    
    def setup(self) -> bool:
        """Setup method for compatibility."""
        return True
    
    def create_embeddings(self, texts: List[str]) -> bool:
        """Create embeddings for texts."""
        return self.add_documents(texts)
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[str]:
        """Search for similar documents and return just the text."""
        results = self.similarity_search(query, k)
        return [doc for doc, score, metadata in results] 
