import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

class ComplaintChunker:
    """Handle text chunking for complaint narratives."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if not text or pd.isna(text):
            return []
        return self.text_splitter.split_text(text)


class ComplaintEmbedder:
    """Handle embedding generation for complaint chunks."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Loaded embedding model: {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        return self.model.encode(texts, show_progress_bar=True)


class VectorStoreManager:
    """Manage ChromaDB vector store for complaint chunks."""
    
    def __init__(self, persist_directory: str = '../vector_store'):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None
    
    def create_collection(self, name: str = "complaint_chunks"):
        """Create or get collection."""
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"description": "CFPB complaint chunks with embeddings"}
        )
        print(f"Collection '{name}' ready")
    
    def add_chunks(self, chunks: List[str], embeddings: np.ndarray, 
                   metadata_list: List[Dict], ids: List[str]):
        """Add chunks to vector store."""
        self.collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadata_list,
            ids=ids
        )
        print(f"Added {len(chunks)} chunks to vector store")
    
    def search(self, query_embedding: np.ndarray, n_results: int = 5):
        """Search for similar chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        return results


def create_stratified_sample(df: pd.DataFrame, sample_size: int = 12000, 
                            random_state: int = 42) -> pd.DataFrame:
    """Create stratified sample across product categories."""
    # Group by product
    grouped = df.groupby('Product')
    
    # Calculate sample size per product
    samples = []
    for product, group in grouped:
        n_samples = int(len(group) / len(df) * sample_size)
        # Ensure at least some samples per product
        n_samples = max(n_samples, min(len(group), 1000))
        sample = group.sample(n=min(n_samples, len(group)), random_state=random_state)
        samples.append(sample)
    
    return pd.concat(samples, ignore_index=True)


def main():
    """Main pipeline for chunking, embedding, and indexing."""
    print("=== Task 2: Text Chunking, Embedding, and Vector Store ===\n")
    
    # Load filtered dataset
    data_path = Path('../data/processed/filtered_complaints.csv')
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(df)} records")
    
    # Create stratified sample
    print("\n--- Creating Stratified Sample ---")
    df_sample = create_stratified_sample(df, sample_size=12000)
    print(f"Sample size: {len(df_sample)}")
    print("\nSample distribution:")
    print(df_sample['Product'].value_counts())
    
    # Initialize chunker
    print("\n--- Text Chunking ---")
    chunker = ComplaintChunker(chunk_size=500, chunk_overlap=50)
    
    # Chunk all narratives
    all_chunks = []
    all_metadata = []
    all_ids = []
    
    for idx, row in df_sample.iterrows():
        chunks = chunker.chunk_text(row['cleaned_narrative'])
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                'complaint_id': row.get('Complaint ID', f'comp_{idx}'),
                'product_category': row['Product'],
                'product': row.get('Product', 'Unknown'),
                'issue': row.get('Issue', 'Unknown'),
                'sub_issue': row.get('Sub-issue', ''),
                'company': row.get('Company', 'Unknown'),
                'state': row.get('State', ''),
                'date_received': row.get('Date received', ''),
                'chunk_index': chunk_idx,
                'total_chunks': len(chunks)
            })
            all_ids.append(f"{idx}_{chunk_idx}")
    
    print(f"Total chunks created: {len(all_chunks)}")
    
    # Initialize embedder
    print("\n--- Generating Embeddings ---")
    embedder = ComplaintEmbedder('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = embedder.embed_texts(all_chunks)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Create vector store
    print("\n--- Creating Vector Store ---")
    vector_store = VectorStoreManager('../vector_store')
    vector_store.create_collection("complaint_chunks_sample")
    vector_store.add_chunks(all_chunks, embeddings, all_metadata, all_ids)
    
    # Save metadata for reference
    metadata = {
        'total_chunks': len(all_chunks),
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'chunk_size': 500,
        'chunk_overlap': 50,
        'sample_size': len(df_sample)
    }
    
    metadata_path = Path('../vector_store/metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n=== Pipeline Complete ===")
    print(f"Vector store saved to: {vector_store.persist_directory}")
    print(f"Total chunks indexed: {len(all_chunks)}")


if __name__ == "__main__":
    main()