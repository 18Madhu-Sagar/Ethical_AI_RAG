# 🤖 Ethical AI RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for AI ethics documents. This system allows you to upload PDF documents about AI ethics and ask intelligent questions with source-backed answers.

## ✨ Features

- **📄 Multi-PDF Processing**: Upload and process multiple PDF documents simultaneously
- **🧠 Vector Search**: Advanced semantic search using embeddings and vector databases
- **💡 Multiple Query Types**: 
  - Simple Q&A
  - Comprehensive answers with source aggregation
  - Source comparison across documents
  - Keyword search with relevance scoring
- **🎯 Response Refinement**: AI-powered response cleaning and summarization
- **🌐 Web Interface**: Beautiful Streamlit web application
- **📊 Analytics**: Document statistics and query history
- **⚙️ Modular Design**: Clean, extensible codebase

## 🏗️ Architecture

The system is built with a modular architecture:

```
├── pdf_extractor.py      # PDF text extraction with fallback methods
├── document_processor.py # Text chunking and preprocessing
├── vector_store.py       # Vector embeddings and similarity search
├── response_refiner.py   # AI-powered response improvement
├── rag_system.py         # Main system orchestration
├── streamlit_app.py      # Web interface
├── main.py              # CLI interface
└── requirements.txt     # Dependencies
```

## 🚀 Quick Start

### Option 1: Streamlit Web Interface (Recommended)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Web Application**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Upload and Query**
   - Open your browser to the displayed URL (usually http://localhost:8501)
   - Upload AI ethics PDF documents using the sidebar
   - Click "Process Documents"
   - Start asking questions!

### Option 2: Command Line Interface

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place PDF Files**
   ```bash
   # Place your AI ethics PDF files in the current directory
   cp /path/to/your/pdfs/*.pdf .
   ```

3. **Run the System**
   ```bash
   # Interactive mode
   python main.py
   
   # Setup only
   python main.py --setup
   
   # Single query
   python main.py --query "What are the main ethical principles of AI?"
   
   # Demo queries
   python main.py --demo
   
   # Show system info
   python main.py --info
   ```

## 📖 Usage Examples

### Web Interface Usage

1. **Upload Documents**: Use the sidebar to upload multiple PDF files
2. **Process Documents**: Click the "Process Documents" button
3. **Ask Questions**: Choose from different query types:
   - **Simple Question**: Get direct answers with source citations
   - **Comprehensive Answer**: Get detailed responses combining multiple sources
   - **Compare Sources**: See how different documents address the same topic
   - **Keyword Search**: Find content with relevance scores

### CLI Usage Examples

```bash
# Basic setup and interactive session
python main.py

# Use a specific directory for PDFs
python main.py --dir /path/to/ethics/pdfs

# Force rebuild of vector database
python main.py --force-rebuild

# Disable response refinement for faster responses
python main.py --no-refinement

# Run demo queries to see system capabilities
python main.py --demo
```

### Python API Usage

```python
from rag_system import EthicalAIRAG

# Initialize the system
rag = EthicalAIRAG(
    pdf_directory="./ethics_pdfs",
    use_refinement=True
)

# Setup (one-time process)
success = rag.setup()

if success:
    # Ask questions
    results = rag.ask_ethics_question("What is algorithmic bias?")
    
    # Get comprehensive answers
    answer = rag.get_comprehensive_answer("How should AI be regulated?", "medium")
    
    # Compare sources
    sources = rag.compare_sources("AI transparency")
    
    # Keyword search
    keyword_results = rag.search_keywords("fairness accountability", num_results=5)
```

## 🔧 Configuration

### Environment Variables

- `HF_TOKEN`: Hugging Face API token (optional, for private models)

### System Parameters

You can customize the system behavior by modifying parameters in the classes:

```python
# Document chunking
DocumentProcessor(chunk_size=1000, chunk_overlap=200)

# Vector store
VectorStore(model_name="all-MiniLM-L6-v2")

# Response refinement
ResponseRefiner(use_summarizer=True)  # Enable AI summarization
```

## 📊 Supported Document Types

The system works best with:

- **Academic Papers**: Research on AI ethics, fairness, transparency
- **Policy Documents**: Government AI guidelines and regulations
- **Ethics Guidelines**: Corporate AI ethics frameworks
- **Technical Reports**: AI bias studies, algorithmic auditing reports
- **Standards Documents**: IEEE, ISO standards for AI systems

## 🧠 Technical Details

### PDF Processing
- **Primary**: PyPDF2 for text extraction
- **Fallback**: Configurable for additional extraction methods
- **Cleaning**: Automatic text normalization and cleanup

### Vector Embeddings
- **Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Database**: ChromaDB with persistence
- **Search**: Cosine similarity with configurable k-values

### Response Refinement
- **Rule-based**: Pattern removal, deduplication
- **AI-powered**: Optional BART-based summarization
- **Length Control**: Short/medium/long response options

### Query Types
1. **Simple Q&A**: Direct similarity search with refinement
2. **Comprehensive**: Multi-source aggregation with synthesis
3. **Source Comparison**: Grouped results by document source
4. **Keyword Search**: Relevance-scored term matching

## 🚨 Troubleshooting

### Common Issues

**PDF Extraction Fails**
```bash
# Check if PDFs are text-based (not scanned images)
# Ensure PDFs are not password-protected
# Try with different PDF files
```

**Vector Store Creation Fails**
```bash
# Check available memory (embeddings require RAM)
# Ensure write permissions in the directory
# Try with fewer/smaller documents initially
```

**Slow Performance**
```bash
# Use CPU-only mode: export CUDA_VISIBLE_DEVICES=""
# Reduce chunk size in DocumentProcessor
# Disable response refinement: use_refinement=False
```

**Streamlit Issues**
```bash
# Clear Streamlit cache
streamlit cache clear

# Check port availability
streamlit run streamlit_app.py --server.port 8502
```

### Memory Requirements

- **Minimum**: 8GB RAM for small document sets
- **Recommended**: 16GB RAM for multiple large documents
- **GPU**: Optional, will use CUDA if available

## 📈 Performance Tips

1. **Document Optimization**
   - Use text-based PDFs (not scanned images)
   - Remove very large, non-relevant documents
   - Consider document preprocessing for better extraction

2. **System Tuning**
   - Adjust chunk size based on document structure
   - Use appropriate embedding models for your domain
   - Enable/disable refinement based on speed vs. quality needs

3. **Query Optimization**
   - Use specific, focused queries for better results
   - Try different query types for different needs
   - Use keyword search for exact term matching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: Framework for building applications with LLMs
- **ChromaDB**: Vector database for embeddings
- **Streamlit**: Web application framework
- **Hugging Face**: Transformers and embedding models
- **PyPDF2**: PDF text extraction

## 📧 Support

If you encounter issues or have questions:

1. Check the troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Include system information and error logs

## 🎯 Roadmap

- [ ] **OCR Support**: Add optical character recognition for scanned PDFs
- [ ] **More Formats**: Support for DOCX, TXT, and other document formats
- [ ] **Advanced Analytics**: Query analytics and document insights
- [ ] **API Endpoints**: REST API for programmatic access
- [ ] **Cloud Deployment**: One-click deployment options
- [ ] **Multi-language**: Support for non-English documents
- [ ] **Custom Models**: Fine-tuned embeddings for ethics domain

---

**Built with ❤️ for the AI Ethics community** 