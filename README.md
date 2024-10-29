# Python OpenAI RAG Implementation

A robust and flexible Retrieval Augmented Generation (RAG) implementation using Python and OpenAI's APIs. This script provides a comprehensive solution for processing documents, generating embeddings, and performing semantic search with GPT-based response generation.

## 🌟 Features

- Multiple chunking strategies (Fixed Size, Semantic, Sentence, Paragraph)
- Async processing for improved performance
- Embedding caching system
- PDF document support
- Configurable model parameters
- Error handling with exponential backoff
- Token counting and management
- Detailed logging
- Type hints for better code clarity

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key

## 🛠️ Installation

1. Clone the repository:
   ```
   git clone https://github.com/rasha-rahman123/python-openai-rag-chat-with-pdf.git
   cd python-openai-rag-chat-with-pdf
   ```

2. Install required packages:
   ```
   pip install openai numpy tiktoken PyPDF2 aiohttp backoff
   ```

3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## 🏗️ Architecture

### Core Components

1. **Document Class**
   - Represents a document with content and metadata
   - Includes fields for content, metadata, document ID, timestamp, and source

2. **ChunkingStrategy Enum**
   - FIXED_SIZE: Chunks by token count
   - SEMANTIC: (Placeholder for semantic-based chunking)
   - SENTENCE: (Placeholder for sentence-based chunking)
   - PARAGRAPH: (Placeholder for paragraph-based chunking)

3. **RAGProcessor Class**
   - Main class handling all RAG operations
   - Configurable parameters for model, chunking, and processing

## 🔧 Configuration Options

```python
RAGProcessor(
    openai_api_key: str,
    model: str = "gpt-4",                    # Main LLM model
    embedding_model: str = "text-embedding-3-small",  # Embedding model
    chunk_size: int = 500,                   # Tokens per chunk
    chunk_overlap: int = 50,                 # Overlap between chunks
    max_tokens: int = 4000,                  # Max response tokens
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
    temperature: float = 0.7,                # Response creativity
    cache_dir: Optional[str] = None          # Embedding cache directory
)
```

## 💻 Usage

### Basic Usage

```python
import asyncio
from rag_processor import RAGProcessor, Document
import os

async def main():
    # Initialize processor
    processor = RAGProcessor(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        cache_dir="embeddings_cache"
    )
    
    # Process a PDF document
    pdf_path = "your_document.pdf"
    pdf_text = convert_pdf_to_text(pdf_path)
    
    # Create document
    doc = Document(
        content=pdf_text,
        doc_id="doc1",
        metadata={"source": pdf_path}
    )
    
    # Process document
    chunks = processor.chunk_document(doc)
    embedded_chunks = await processor.embed_chunks(chunks)
    
    # Save embeddings for future use
    processor.save_embeddings(embedded_chunks, "my_embeddings.json")
    
    # Query the system
    query = "Your question here?"
    result = await processor.process_query(query, embedded_chunks)
    print(result['response'])

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔍 Key Features Explained

### 1. Document Chunking
The system supports multiple chunking strategies with the default being FIXED_SIZE:
- Splits documents into manageable chunks
- Maintains context with configurable overlap
- Preserves document metadata across chunks

### 2. Embedding Generation
- Uses OpenAI's embedding models
- Async processing for better performance
- Includes caching system for efficiency

### 3. Semantic Search
- Calculates similarity between query and chunks
- Returns top-k most relevant chunks
- Uses dot product similarity scoring

### 4. Response Generation
- Contextual response generation using GPT models
- Configurable system prompts
- Citation support for sources

## 💾 Caching System

The system includes a built-in caching mechanism for embeddings:

```python
# Save embeddings
processor.save_embeddings(embedded_chunks, "embeddings_file.json")

# Load embeddings
embedded_chunks = processor.load_embeddings("embeddings_file.json")
```

## 🔄 Error Handling

- Implements exponential backoff for API calls
- Comprehensive error logging
- Graceful failure handling

## 📊 Performance Considerations

- Async processing for improved throughput
- Token counting to optimize chunk sizes
- Embedding caching to reduce API calls
- Configurable chunk overlap for context preservation

## 🛡️ Best Practices

1. **API Key Security**
   - Always use environment variables
   - Never commit API keys to version control

2. **Resource Management**
   - Cache embeddings when possible
   - Monitor token usage
   - Use appropriate chunk sizes

3. **Error Handling**
   - Implement proper error handling
   - Monitor API rate limits
   - Use backoff strategies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for their API and models
- The Python community for the amazing libraries

## 📞 Support

For issues and feature requests, please use the GitHub issues page.
