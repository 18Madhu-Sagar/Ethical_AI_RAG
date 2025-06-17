# 🚀 Deployment Guide - Simplified Ethical AI RAG System

This guide will help you deploy the **working** Simplified Ethical AI RAG System.

## ✅ System Status

**WORKING** ✅ - This system is tested and functional with Python 3.13

## 📋 Prerequisites

- **Python 3.13+** (tested and working)
- **2GB RAM minimum**
- **100MB disk space** for dependencies
- **Internet connection** for initial package installation

## 🔧 Installation Steps

### 1. Clone or Download
```bash
# If using git
git clone <repository-url>
cd EthicalAI

# Or download and extract the ZIP file
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- `streamlit` - Web interface
- `PyPDF2` - PDF processing
- `python-docx` - Document processing
- `scikit-learn` - TF-IDF vectorization
- `numpy` - Numerical operations
- `nltk` - Text processing (optional)
- `python-dotenv` - Environment variables

### 3. Verify Installation
```bash
python -c "import streamlit, sklearn, PyPDF2; print('✅ All dependencies installed')"
```

## 🚀 Running the Application

### Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```

**Access the app at:** http://localhost:8501

### Command Line Interface
```bash
python main.py
```

## 🧪 Testing the System

### 1. Upload Test Document
- Use the included `AI_Ethics_Sample.pdf`
- Or upload your own PDF document

### 2. Test Questions
Try these sample questions:
- "What are the main principles of AI ethics?"
- "How can we ensure AI fairness?"
- "What are the risks of biased AI systems?"

### 3. Verify Results
- Check that documents are processed successfully
- Verify that search returns relevant results
- Confirm that answers combine multiple sources

## 🔍 System Architecture

```
User Upload → PDF Extraction → Text Chunking → TF-IDF Vectors → Search → Answer
     ↓              ↓              ↓              ↓           ↓        ↓
  Streamlit    PyPDF2    DocumentProcessor  VectorStore  Similarity  RAGSystem
```

## 📊 Performance Expectations

### Processing Times
- **Small PDF (1-5 pages)**: 5-10 seconds
- **Medium PDF (10-20 pages)**: 15-30 seconds
- **Large PDF (50+ pages)**: 1-2 minutes

### Query Response
- **Search time**: < 1 second
- **Answer generation**: < 2 seconds

### Memory Usage
- **Base system**: ~200MB
- **Per document**: ~10-50MB depending on size

## 🛠️ Configuration Options

### Environment Variables (Optional)
Create a `.env` file:
```bash
# Chunk size for document processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Number of search results to return
MAX_SEARCH_RESULTS=5

# TF-IDF parameters
TFIDF_MAX_FEATURES=5000
TFIDF_MIN_DF=1
TFIDF_MAX_DF=0.95
```

### Customization
- **Chunk sizes**: Edit `document_processor.py`
- **TF-IDF parameters**: Edit `vector_store.py`
- **UI appearance**: Edit `streamlit_app.py`

## 🔧 Troubleshooting

### Common Issues

#### "ModuleNotFoundError"
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### "PDF processing failed"
- Ensure PDF is not password-protected
- Check file size (max recommended: 50MB)
- Verify file is not corrupted

#### "Vector store not ready"
- Upload a document first
- Wait for processing to complete
- Check system status in sidebar

#### "Port already in use"
```bash
# Use different port
streamlit run streamlit_app.py --server.port 8502
```

### Performance Issues

#### Slow processing
- Reduce chunk size in configuration
- Use smaller PDF files for testing
- Check available RAM

#### High memory usage
- Clear vector store between documents
- Restart the application periodically
- Use smaller TF-IDF feature count

## 🌐 Production Deployment

### Local Network Access
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

### Docker Deployment (Optional)
```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]
```

### Cloud Deployment
- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Use provided `runtime.txt`
- **AWS/GCP**: Standard Python app deployment

## 📈 Monitoring

### System Health Checks
- Monitor memory usage
- Check processing times
- Verify search accuracy

### Logs
- Application logs in terminal
- Streamlit logs in browser console
- Error tracking in system status

## 🔄 Updates and Maintenance

### Regular Maintenance
- Clear old vector stores periodically
- Update dependencies monthly
- Monitor disk space usage

### Backup Important Data
- Vector store data in `simple_vector_db/`
- Uploaded documents (if needed)
- Configuration files

## 📞 Support

### Self-Help
1. Check this deployment guide
2. Review error messages in terminal
3. Check system status in web interface
4. Try with sample document first

### Getting Help
1. Check the main README.md
2. Review code comments
3. Create an issue on the repository

---

**This system is designed to work reliably** - If you follow this guide, you should have a working RAG system in under 10 minutes. 