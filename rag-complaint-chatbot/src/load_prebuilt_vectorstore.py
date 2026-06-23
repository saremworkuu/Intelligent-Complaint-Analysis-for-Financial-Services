import pandas as pd
import numpy as np
from pathlib import Path
import chromadb
from chromadb.config import Settings


def load_prebuilt_embeddings(parquet_path: str) -> pd.DataFrame:
    """Load pre-built embeddings from parquet file."""
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} chunks from parquet file")
    return df


def create_chroma_from_parquet(parquet_path: str, output_path: str):
    """Create ChromaDB collection from pre-built parquet embeddings."""
    # Load data
    df = load_prebuilt_embeddings(parquet_path)
    
    # Initialize ChromaDB
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=str(output_dir),
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection = client.get_or_create_collection(
        name="complaint_chunks_full",
        metadata={"description": "Full CFPB complaint chunks from pre-built embeddings"}
    )
    
    # Extract data
    texts = df['text'].tolist()
    embeddings = df['embedding'].tolist()
    
    # Build metadata list
    metadata_list = []
    for _, row in df.iterrows():
        metadata = {
            'complaint_id': row.get('complaint_id', ''),
            'product_category': row.get('product_category', ''),
            'product': row.get('product', ''),
            'issue': row.get('issue', ''),
            'sub_issue': row.get('sub_issue', ''),
            'company': row.get('company', ''),
            'state': row.get('state', ''),
            'date_received': row.get('date_received', ''),
            'chunk_index': row.get('chunk_index', 0),
            'total_chunks': row.get('total_chunks', 1)
        }
        metadata_list.append(metadata)
    
    # Create IDs
    ids = [f"chunk_{i}" for i in range(len(df))]
    
    # Add to collection in batches
    batch_size = 1000
    for i in range(0, len(df), batch_size):
        end_idx = min(i + batch_size, len(df))
        collection.add(
            documents=texts[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=metadata_list[i:end_idx],
            ids=ids[i:end_idx]
        )
        print(f"Added chunks {i} to {end_idx}")
    
    print(f"\nCreated ChromaDB collection with {len(df)} chunks")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parquet_path = '../data/raw/complaint_embeddings.parquet'
    output_path = '../vector_store'
    create_chroma_from_parquet(parquet_path, output_path)