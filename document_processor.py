import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DocumentProcessor:
    """Process and chunk documents for RAG system."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove multiple newlines and spaces
        cleaned_text = re.sub(r'\n+', '\n', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        return cleaned_text
    
    def create_rag_documents(self, documents_text: Dict[str, str]) -> List[Document]:
        """Create RAG document chunks from extracted text."""
        print("\n" + "="*50)
        print("🔧 CREATING RAG DOCUMENT CHUNKS")
        print("="*50)
        
        all_rag_documents = []
        
        for source, text in documents_text.items():
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
            
            print(f"📄 {source}: {len(valid_chunks)} chunks created")
            
            # Create Document objects
            for i, chunk in enumerate(valid_chunks):
                doc = Document(
                    page_content=chunk.strip(),
                    metadata={
                        "source": source.replace('.pdf', ''),
                        "chunk_id": i,
                        "total_chunks": len(valid_chunks)
                    }
                )
                all_rag_documents.append(doc)
        
        print(f"\n✅ Total RAG documents created: {len(all_rag_documents)}")
        
        if len(all_rag_documents) == 0:
            print("❌ No valid document chunks created!")
            return []
        
        return all_rag_documents
    
    def get_chunk_statistics(self, documents: List[Document]) -> Dict:
        """Get statistics about document chunks."""
        if not documents:
            return {}
        
        chunk_lengths = [len(doc.page_content) for doc in documents]
        sources = [doc.metadata.get('source', 'unknown') for doc in documents]
        
        stats = {
            'total_chunks': len(documents),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'sources': list(set(sources)),
            'chunks_per_source': {source: sources.count(source) for source in set(sources)}
        }
        
        return stats
    
    def preview_chunks(self, documents: List[Document], num_samples: int = 3) -> None:
        """Preview a few document chunks."""
        print(f"\n📋 PREVIEW OF {min(num_samples, len(documents))} CHUNKS:")
        print("="*50)
        
        for i, doc in enumerate(documents[:num_samples]):
            source = doc.metadata.get('source', 'unknown')
            chunk_id = doc.metadata.get('chunk_id', 'unknown')
            preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            
            print(f"\n🔍 Chunk {i+1} from {source} (ID: {chunk_id}):")
            print(f"   {preview}")


if __name__ == "__main__":
    # Example usage
    from pdf_extractor import PDFExtractor
    
    print("Starting document processing...")
    
    # Extract text from PDFs
    extractor = PDFExtractor()
    documents_text = extractor.extract_from_directory()
    
    if documents_text:
        # Process documents
        processor = DocumentProcessor()
        rag_documents = processor.create_rag_documents(documents_text)
        
        if rag_documents:
            # Show statistics
            stats = processor.get_chunk_statistics(rag_documents)
            print("\n" + "="*50)
            print("📊 DOCUMENT PROCESSING STATISTICS")
            print("="*50)
            print(f"Total chunks: {stats['total_chunks']}")
            print(f"Average chunk length: {stats['avg_chunk_length']:.1f} characters")
            print(f"Min/Max chunk length: {stats['min_chunk_length']}/{stats['max_chunk_length']}")
            print(f"Sources: {', '.join(stats['sources'])}")
            
            for source, count in stats['chunks_per_source'].items():
                print(f"  {source}: {count} chunks")
            
            # Preview some chunks
            processor.preview_chunks(rag_documents)
    else:
        print("No documents to process.") 