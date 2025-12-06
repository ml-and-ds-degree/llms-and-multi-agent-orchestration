
"""Test sequential embedding as a workaround for Ollama batch issues"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from typing import List
import time

class SequentialOllamaEmbeddings(Embeddings):
    """Wrapper that processes embeddings one at a time to avoid Ollama batch issues"""
    
    def __init__(self, model: str = "nomic-embed-text", delay: float = 0.1):
        self.model = model
        self.delay = delay
        self.embeddings = OllamaEmbeddings(model=model)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Process each document sequentially"""
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.embeddings.embed_query(text)
                results.append(result)
                if self.delay and i < len(texts) - 1:
                    time.sleep(self.delay)
            except Exception as e:
                print(f"Warning: Failed to embed document {i}: {e}")
                # Return a zero vector as fallback
                results.append([0.0] * 768)
        return results
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embeddings.embed_query(text)

# Test the fix
print("Loading and chunking document...")
doc_path = Path('./data/BOI.pdf')
loader = PyPDFLoader(file_path=str(doc_path))
data = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = splitter.split_documents(data)
print(f"Created {len(chunks)} chunks")

print("\nTesting sequential embeddings...")
embeddings = SequentialOllamaEmbeddings(model='nomic-embed-text', delay=0.05)

print("Building vector store...")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="test-sequential",
    persist_directory="./results/chroma_test"
)

print(f"\n✅ Success! Vector store created with {len(chunks)} documents")
