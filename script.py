import os
import json
import numpy as np
import logging
import asyncio
import aiohttp
from typing import List, Dict, Union, Optional
from pathlib import Path
from datetime import datetime
from openai import AsyncOpenAI, OpenAI
import tiktoken
from concurrent.futures import ThreadPoolExecutor
import backoff
from dataclasses import dataclass
from enum import Enum
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"

@dataclass
class Document:
    """Represents a document with its content and metadata."""
    content: str
    metadata: Dict = None
    doc_id: str = None
    timestamp: Optional[str] = None
    source: Optional[str] = None

class RAGProcessor:
    """
    A generic RAG (Retrieval Augmented Generation) processor that can handle various document types
    and implement different chunking and embedding strategies.
    """
    
    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        max_tokens: int = 4000,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
        temperature: float = 0.7,
        cache_dir: Optional[str] = None
    ):
        """Initialize the RAG processor with configuration parameters."""
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.sync_client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.chunking_strategy = chunking_strategy
        self.temperature = temperature
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.tokenizer = tiktoken.encoding_for_model(model)

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk a document based on the selected strategy.
        Returns a list of Document objects, each representing a chunk.
        """
        if self.chunking_strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_by_tokens(document)
        elif self.chunking_strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(document)
        elif self.chunking_strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_by_sentence(document)
        elif self.chunking_strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(document)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunking_strategy}")

    def _chunk_by_tokens(self, document: Document) -> List[Document]:
        """Chunk document by token count with overlap."""
        chunks = []
        content = document.content
        current_chunk = []
        current_size = 0
        
        words = content.split()
        
        for i, word in enumerate(words):
            word_tokens = self._count_tokens(word)
            if current_size + word_tokens > self.chunk_size and current_chunk:
                # Create new chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(Document(
                    content=chunk_text,
                    metadata=document.metadata,
                    doc_id=f"{document.doc_id}_chunk_{len(chunks)}",
                    source=document.source
                ))
                
                # Handle overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_size = self._count_tokens(" ".join(current_chunk))
            
            current_chunk.append(word)
            current_size += word_tokens
        
        # Add final chunk if it exists
        if current_chunk:
            chunks.append(Document(
                content=" ".join(current_chunk),
                metadata=document.metadata,
                doc_id=f"{document.doc_id}_chunk_{len(chunks)}",
                source=document.source
            ))
        
        return chunks

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3
    )
    async def get_embedding(self, text: str) -> List[float]:
        """Get embeddings for a text string using OpenAI's API."""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    async def embed_chunks(self, chunks: List[Document]) -> Dict[str, Dict]:
        """
        Embed all chunks and return a dictionary mapping chunk IDs to their embeddings and content.
        """
        embedded_chunks = {}
        
        for chunk in chunks:
            embedding = await self.get_embedding(chunk.content)
            embedded_chunks[chunk.doc_id] = {
                "embedding": embedding,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "source": chunk.source,
                "timestamp": chunk.timestamp
            }
            
        return embedded_chunks

    def semantic_search(
        self,
        query: str,
        embedded_chunks: Dict[str, Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Perform semantic search to find the most relevant chunks for a query.
        """
        # Get query embedding
        query_embedding = self.sync_client.embeddings.create(
            model=self.embedding_model,
            input=query
        ).data[0].embedding

        # Calculate similarities
        similarities = []
        for chunk_id, chunk_data in embedded_chunks.items():
            similarity = np.dot(query_embedding, chunk_data["embedding"])
            similarities.append((chunk_id, similarity))

        # Sort by similarity and get top_k results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_chunks = []
        
        for chunk_id, similarity in similarities[:top_k]:
            chunk_data = embedded_chunks[chunk_id]
            top_chunks.append({
                "content": chunk_data["content"],
                "metadata": chunk_data["metadata"],
                "similarity": similarity,
                "source": chunk_data["source"],
                "timestamp": chunk_data["timestamp"]
            })

        return top_chunks

    async def generate_response(
        self,
        query: str,
        relevant_chunks: List[Dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response using the relevant chunks and query.
        """
        # Build context from relevant chunks
        context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that provides accurate, 
            well-structured answers based on the provided context. Always cite your sources
            when referring to specific information."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Context:\n{context}\n\nQuestion: {query}
            
            Please provide a clear, well-structured answer based on the context provided.
            Include relevant citations when appropriate."""}
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    async def process_query(
        self,
        query: str,
        embedded_chunks: Dict[str, Dict],
        top_k: int = 5,
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Process a query end-to-end: semantic search + response generation.
        """
        # Get relevant chunks
        relevant_chunks = self.semantic_search(query, embedded_chunks, top_k)
        
        # Generate response
        response = await self.generate_response(query, relevant_chunks, system_prompt)
        
        return {
            "query": query,
            "response": response,
            "relevant_chunks": relevant_chunks
        }

    def save_embeddings(self, embedded_chunks: Dict[str, Dict], filename: str):
        """Save embeddings to disk."""
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.cache_dir / filename
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_chunks = {}
            for chunk_id, chunk_data in embedded_chunks.items():
                embedding = chunk_data["embedding"]
                # Convert to list if it's a numpy array, otherwise use as is
                embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                
                serializable_chunks[chunk_id] = {
                    **chunk_data,
                    "embedding": embedding_list
                }
            
            with open(filepath, 'w') as f:
                json.dump(serializable_chunks, f)

    def load_embeddings(self, filename: str) -> Optional[Dict[str, Dict]]:
        """Load embeddings from disk."""
        if self.cache_dir:
            filepath = self.cache_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # Convert lists back to numpy arrays
                    for chunk_id in data:
                        data[chunk_id]["embedding"] = np.array(data[chunk_id]["embedding"])
                    return data
        return None

# Example usage
async def main():
    # Initialize processor
    processor = RAGProcessor(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        cache_dir="embeddings_cache"
    )
    # Convert pdf to text
    pdf_to_text = PyPDF2.PdfReader(open("consciousness.pdf", "rb"))
    text = ""
    for page in pdf_to_text.pages:
        text += page.extract_text()
    
    # Example document
    doc = Document(
        content=text,
        doc_id="doc1",
        metadata={"source": "consciousness.pdf"},
        timestamp=datetime.now().isoformat()
    )
    
    # Process document
    chunks = processor.chunk_document(doc)
    embedded_chunks = await processor.embed_chunks(chunks)
    
    # Save embeddings
    processor.save_embeddings(embedded_chunks, "example_embeddings.json")
    
    # Process query
    query = "What is consciousness?"
    result = await processor.process_query(query, embedded_chunks)
    print(f"Response: {result['response']}")

if __name__ == "__main__":
    asyncio.run(main())
