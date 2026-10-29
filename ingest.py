"""
RAG Pipeline - Data Ingestion Script
Reads kb_chunks.json, generates embeddings using NLP, and builds FAISS index
Implements the indexing phase of the RAG (Retrieval-Augmented Generation) pipeline
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
DATASET_PATH = "kb_chunks.json"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32
OUTPUT_DIR = "data/chatbot_index"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_dataset(file_path):
    """Load the knowledge base chunks from JSON file"""
    print(f"Loading dataset from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}\n"
            "Please ensure kb_chunks.json exists in the current directory."
        )
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        chunks = data
    elif isinstance(data, dict) and 'chunks' in data:
        chunks = data['chunks']
    elif isinstance(data, dict) and 'data' in data:
        chunks = data['data']
    else:
        raise ValueError("Unsupported JSON structure. Expected list or dict with 'chunks' or 'data' key.")
    
    print(f"Loaded {len(chunks)} chunks from dataset")
    return chunks

def extract_text_and_metadata(chunks):
    """Extract text, source, and metadata from chunks"""
    texts = []
    sources = []
    metadata_list = []
    
    for i, chunk in enumerate(chunks):
        # Extract text
        if isinstance(chunk, dict):
            text = chunk.get('text', chunk.get('content', str(chunk)))
            source = chunk.get('source', chunk.get('title', f"chunk_{i}"))
            metadata = chunk.get('metadata', {})
            
            # If metadata is empty, create basic metadata
            if not metadata:
                metadata = {
                    'index': i,
                    'source': source,
                    'chunk_id': i
                }
        else:
            # If chunk is just a string
            text = str(chunk)
            source = f"chunk_{i}"
            metadata = {'index': i, 'chunk_id': i}
        
        # Store text in metadata for retrieval in chat.py
        metadata['text'] = text
        
        texts.append(text)
        sources.append(source)
        metadata_list.append(metadata)
    
    return texts, sources, metadata_list

def generate_embeddings(texts, model, batch_size=32):
    """Generate embeddings for all texts using batching"""
    print(f"Generating embeddings using {MODEL_NAME}...")
    
    embeddings = []
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings
    embeddings = np.vstack(embeddings).astype('float32')
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    return embeddings

def build_faiss_index(embeddings):
    """Build FAISS index for similarity search"""
    print("Building FAISS index...")
    
    dimension = embeddings.shape[1]
    
    # Use IndexFlatIP (Inner Product) for cosine similarity with normalized embeddings
    # Since embeddings are normalized, inner product = cosine similarity
    index = faiss.IndexFlatIP(dimension)
    
    # Add embeddings to index
    index.add(embeddings)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    return index

def save_index_and_metadata(index, embeddings, metadata_list, sources, output_dir):
    """Save FAISS index, embeddings, and metadata"""
    print("Saving index and metadata...")
    
    # Save FAISS index
    index_path = os.path.join(output_dir, "faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    metadata_dict = {
        'metadata_list': metadata_list,
        'sources': sources,
        'num_chunks': len(metadata_list),
        'embedding_dim': embeddings.shape[1]
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Saved metadata to {metadata_path}")
    print("\n✅ Ingestion complete!")

def main():
    """Main ingestion pipeline"""
    print("=" * 60)
    print("RAG Pipeline - Data Ingestion")
    print("Building knowledge base with NLP embeddings")
    print("=" * 60)
    
    # Load dataset
    chunks = load_dataset(DATASET_PATH)
    
    # Extract text and metadata
    texts, sources, metadata_list = extract_text_and_metadata(chunks)
    
    # Load sentence transformer model
    print(f"\nLoading SentenceTransformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully")
    
    # Generate embeddings
    embeddings = generate_embeddings(texts, model, BATCH_SIZE)
    
    # Build FAISS index
    index = build_faiss_index(embeddings)
    
    # Save everything
    save_index_and_metadata(index, embeddings, metadata_list, sources, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Ready for chat! Run: python chat.py")
    print("=" * 60)

if __name__ == "__main__":
    main()

