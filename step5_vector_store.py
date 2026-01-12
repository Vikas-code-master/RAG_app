"""
Step 5: Vector Store with ChromaDB
==================================

This script demonstrates how to store embeddings in ChromaDB for persistent
storage and efficient similarity search.

Why Use a Vector Database?
--------------------------
1. **Persistence**: Embeddings are stored on disk, avoiding re-computation
2. **Efficient Search**: Optimized algorithms for similarity search (ANN)
3. **Scalability**: Handle millions of vectors efficiently
4. **Metadata**: Store and filter by document metadata

ChromaDB Features:
------------------
- Lightweight, embedded database (no separate server needed)
- Persistent storage to disk
- Automatic embedding generation (optional)
- Metadata filtering
- Multiple distance metrics (cosine, L2, inner product)
"""

import os
from pathlib import Path
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Default paths
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "book_collection"


def create_vector_store(
    chunks: list,
    embeddings_model,
    persist_directory: str = None,
    collection_name: str = COLLECTION_NAME
):
    """
    Create a ChromaDB vector store from document chunks.
    
    Args:
        chunks: List of Document objects to store
        embeddings_model: Embedding model to use
        persist_directory: Directory to persist the database
        collection_name: Name for the Chroma collection
        
    Returns:
        Chroma vector store instance
    """
    if persist_directory is None:
        persist_directory = str(CHROMA_DIR)
    
    print("=" * 60)
    print("Creating Vector Store")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Collection name: {collection_name}")
    print(f"  - Persist directory: {persist_directory}")
    print(f"  - Number of chunks: {len(chunks)}")
    
    # Create the vector store
    # This will automatically:
    # 1. Generate embeddings for all chunks
    # 2. Store them in ChromaDB
    # 3. Persist to disk
    
    print(f"\nCreating embeddings and storing in ChromaDB...")
    print("This may take a few minutes for large documents.\n")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    
    # Get collection info
    collection = vector_store._collection
    count = collection.count()
    
    print(f"✓ Vector store created successfully!")
    print(f"✓ Stored {count} vectors")
    print(f"✓ Data persisted to: {persist_directory}")
    
    return vector_store


def load_existing_vector_store(
    embeddings_model,
    persist_directory: str = None,
    collection_name: str = COLLECTION_NAME
):
    """
    Load an existing ChromaDB vector store without re-embedding.
    
    This is much faster than recreating the store, as embeddings
    are loaded directly from disk.
    
    Args:
        embeddings_model: Embedding model (needed for queries)
        persist_directory: Directory where DB is stored
        collection_name: Name of the collection to load
        
    Returns:
        Chroma vector store instance
    """
    if persist_directory is None:
        persist_directory = str(CHROMA_DIR)
    
    print("=" * 60)
    print("Loading Existing Vector Store")
    print("=" * 60)
    
    if not Path(persist_directory).exists():
        raise FileNotFoundError(
            f"No vector store found at: {persist_directory}\n"
            "Please create one first using create_vector_store()"
        )
    
    print(f"\nLoading from: {persist_directory}")
    print(f"Collection: {collection_name}")
    
    # Load existing vector store
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings_model,
        collection_name=collection_name,
    )
    
    # Verify loaded data
    collection = vector_store._collection
    count = collection.count()
    
    print(f"\n✓ Loaded successfully!")
    print(f"✓ Found {count} vectors in collection")
    
    return vector_store


def test_vector_store(vector_store, query: str = "What is the main topic of this book?"):
    """
    Test the vector store with a sample query.
    
    Args:
        vector_store: Chroma vector store instance
        query: Test query string
    """
    print("\n" + "=" * 60)
    print("Vector Store Test")
    print("=" * 60)
    
    print(f"\nTest query: \"{query}\"")
    print("\nPerforming similarity search...")
    
    # Perform similarity search
    results = vector_store.similarity_search_with_score(query, k=3)
    
    print(f"\nTop 3 most similar chunks:")
    print("-" * 60)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[Result {i}] Score: {score:.4f}")
        print(f"Source: Page {doc.metadata.get('page', 'N/A') + 1}")
        print(f"Content preview:")
        preview = doc.page_content[:300].replace('\n', ' ')
        if len(doc.page_content) > 300:
            preview += "..."
        print(f"  {preview}")


def get_collection_info(persist_directory: str = None, collection_name: str = COLLECTION_NAME):
    """
    Get information about an existing ChromaDB collection.
    
    Args:
        persist_directory: Directory where DB is stored
        collection_name: Name of the collection
    """
    if persist_directory is None:
        persist_directory = str(CHROMA_DIR)
    
    print("=" * 60)
    print("Collection Information")
    print("=" * 60)
    
    # Use chromadb client directly for metadata
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        collection = client.get_collection(collection_name)
        
        print(f"\nCollection: {collection_name}")
        print(f"  - Vector count: {collection.count()}")
        print(f"  - Metadata: {collection.metadata}")
        
        # Get sample of stored data
        sample = collection.peek(limit=3)
        
        print(f"\nSample entries (first 3):")
        for i, doc_id in enumerate(sample['ids']):
            metadata = sample['metadatas'][i] if sample['metadatas'] else {}
            print(f"  - ID: {doc_id[:20]}... | Page: {metadata.get('page', 'N/A')}")
            
    except Exception as e:
        print(f"Error accessing collection: {e}")


def main():
    """Main function to demonstrate vector store operations."""
    
    # Initialize embeddings model
    print("Initializing Ollama embeddings model...\n")
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
    
    # Check if vector store already exists
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print("Existing vector store found!")
        print("Choose an option:")
        print("  1. Load existing (fast)")
        print("  2. Recreate from PDF (slow)")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            vector_store = load_existing_vector_store(embeddings_model)
            test_vector_store(vector_store)
            return vector_store
    
    # Find PDF file
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        # Check in project root too
        pdf_files = list(PROJECT_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found!")
        print(f"Please add a PDF to: {DATA_DIR}")
        return None
    
    pdf_path = str(pdf_files[0])
    print(f"Using PDF: {pdf_path}\n")
    
    # Load and chunk the document
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    
    print("\nChunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks\n")
    
    # Create vector store
    vector_store = create_vector_store(chunks, embeddings_model)
    
    # Test it
    test_vector_store(vector_store)
    
    # Show collection info
    get_collection_info()
    
    return vector_store


if __name__ == "__main__":
    main()

