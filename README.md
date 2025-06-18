# 🤖 Simplified Ethical AI RAG System

A **reliable and working** RAG (Retrieval-Augmented Generation) system for ethical AI document analysis. This system is designed to work seamlessly with **Python 3.13** and uses proven, stable technologies.

## ✅ What Actually Works

- **TF-IDF Embeddings**: Fast, reliable, and memory-efficient
- **Scikit-learn**: Battle-tested machine learning library
- **PDF Processing**: Robust text extraction from PDF documents
- **Document Chunking**: Intelligent text segmentation
- **Semantic Search**: Cosine similarity-based retrieval
- **Multi-source Answers**: Combines information from multiple document chunks
- **Streamlit Interface**: Clean, modern web interface
- **Python 3.13 Compatible**: Works with the latest Python version

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run streamlit_app.py
```

### 3. Upload and Query
1. Open http://localhost:8501 in your browser
2. Upload a PDF document using the file uploader
3. Wait for processing to complete
4. Ask questions about your document

## 📁 Project Structure

```
EthicalAI/
├── streamlit_app.py          # Main web interface
├── rag_system.py             # Core RAG system logic
├── vector_store.py           # TF-IDF vector storage
├── pdf_extractor.py          # PDF text extraction
├── document_processor.py     # Text chunking and processing
├── response_refiner.py       # Answer refinement utilities
├── main.py                   # CLI interface
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .env.template            # Environment variables template
└── AI_Ethics_Sample.pdf     # Sample document for testing
```

## 🔧 Core Components

### Vector Store (`vector_store.py`)
- **TF-IDF Vectorization**: Uses scikit-learn's TfidfVectorizer
- **Cosine Similarity**: Fast similarity calculations
- **Persistent Storage**: JSON + pickle for data persistence
- **Memory Efficient**: No GPU requirements

### RAG System (`rag_system.py`)
- **Document Processing**: PDF → Chunks → Vectors → Search
- **Multi-source Answers**: Combines top relevant chunks
- **Simple but Effective**: No complex LLM dependencies
- **Reliable**: Works consistently across different environments

### PDF Extractor (`pdf_extractor.py`)
- **Robust Extraction**: Multiple fallback methods
- **Error Handling**: Graceful failure recovery
- **Text Cleaning**: Removes artifacts and formatting issues

## 🎯 Features

### Document Processing
- ✅ PDF text extraction with multiple fallback methods
- ✅ Intelligent text chunking (1000 chars with 200 char overlap)
- ✅ Metadata preservation (source, page numbers, etc.)
- ✅ Batch processing support

### Search & Retrieval
- ✅ TF-IDF semantic search
- ✅ Configurable result count (default: top 5 matches)
- ✅ Similarity scoring
- ✅ Source attribution

### User Interface
- ✅ Clean Streamlit web interface
- ✅ Real-time processing feedback
- ✅ System status monitoring
- ✅ Source document display
- ✅ Clear error messages

## 📊 System Requirements

- **Python**: 3.13+ (tested and working)
- **Memory**: 2GB RAM minimum
- **Storage**: 100MB for dependencies
- **OS**: Windows, macOS, Linux

## 🔍 How It Works

1. **Document Upload**: User uploads PDF through web interface
2. **Text Extraction**: PDF content extracted using PyPDF2
3. **Text Chunking**: Document split into overlapping chunks
4. **Vectorization**: TF-IDF vectors created for all chunks
5. **Storage**: Vectors and metadata persisted to disk
6. **Query Processing**: User question vectorized using same TF-IDF model
7. **Similarity Search**: Cosine similarity finds most relevant chunks
8. **Answer Generation**: Top chunks combined into comprehensive answer

## 🛠️ Configuration

### Environment Variables (Optional)
Create a `.env` file based on `.env.template`:
```bash
# Optional: Customize chunk sizes
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Optional: Customize search results
MAX_SEARCH_RESULTS=5
```

### Customization Options
- **Chunk Size**: Modify in `document_processor.py`
- **TF-IDF Parameters**: Adjust in `vector_store.py`
- **Search Results**: Configure in `rag_system.py`

## 🧪 Testing

### Sample Document
Use the included `AI_Ethics_Sample.pdf` to test the system:

1. Upload the sample PDF
2. Try these questions:
   - "What are the main principles of AI ethics?"
   - "How can we ensure AI fairness?"
   - "What are the risks of biased AI systems?"

### CLI Testing
```bash
python main.py
```

## 🔧 Troubleshooting

### Common Issues

**"No module named 'sklearn'"**
```bash
pip install scikit-learn
```

**"PDF extraction failed"**
- Ensure PDF is not password-protected
- Try a different PDF file
- Check file permissions

**"Vector store not ready"**
- Upload a document first
- Wait for processing to complete
- Check system status in sidebar

### Performance Tips

- **Large PDFs**: System handles up to 50MB PDFs efficiently
- **Memory Usage**: TF-IDF is memory-efficient compared to transformer models
- **Speed**: Initial processing takes 10-30 seconds, queries are instant

## 📈 Advantages of This Approach

### Reliability
- ✅ **No GPU Dependencies**: Runs on any machine
- ✅ **Stable Libraries**: Uses mature, well-tested packages
- ✅ **Python 3.13 Compatible**: Works with latest Python
- ✅ **Consistent Results**: Deterministic behavior

### Performance
- ✅ **Fast Startup**: No model downloading required
- ✅ **Low Memory**: Efficient TF-IDF implementation
- ✅ **Quick Queries**: Sub-second response times
- ✅ **Scalable**: Handles large document collections

### Simplicity
- ✅ **Easy Setup**: Single pip install command
- ✅ **Clear Code**: Well-documented, readable implementation
- ✅ **No External APIs**: Completely self-contained
- ✅ **Minimal Configuration**: Works out of the box

## 🔄 Future Enhancements

While this system is designed to be simple and reliable, potential improvements include:

- **Multiple File Formats**: Support for DOCX, TXT, etc.
- **Advanced Chunking**: Semantic-aware text splitting
- **Query Expansion**: Synonym and related term matching
- **Export Features**: Save results to PDF/Word
- **Batch Upload**: Process multiple files simultaneously

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Create an issue on the repository

---

**Built with reliability in mind** - This system prioritizes working functionality over cutting-edge features. 