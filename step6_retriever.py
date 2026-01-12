"""
Step 6: Create a Retriever
==========================

This script demonstrates how to create a retriever from the ChromaDB vector
store for similarity-based document retrieval.

What is a Retriever?
--------------------
A retriever is an abstraction that takes a query and returns relevant documents.
In RAG, the retriever is responsible for finding context to augment the LLM's
response.

How Retrieval Works:
--------------------
1. **Query Embedding**: The user's question is converted to an embedding vector
2. **Similarity Search**: Find vectors in the database closest to the query
3. **Return Documents**: Return the top-k most similar document chunks

Similarity Metrics:
-------------------
- **Cosine Similarity**: Measures angle between vectors (default in Chroma)
- **Euclidean (L2)**: Measures direct distance
- **Dot Product**: Measures magnitude and direction

Why top-k = 3-5?
----------------
- Too few: May miss relevant context
- Too many: Adds noise and uses token budget
- 3-5 is a good balance for most use cases
"""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings


# Default paths
PROJECT_DIR = Path(__file__).parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "book_collection"


def create_retriever(
    vector_store,
    search_type: str = "similarity",
    k: int = 4
):
    """
    Create a retriever from a vector store.
    
    Args:
        vector_store: ChromaDB vector store instance
        search_type: Type of search ("similarity" or "mmr")
        k: Number of documents to retrieve
        
    Returns:
        Configured retriever
        
    Search Types:
    - similarity: Returns most similar documents
    - mmr (Maximal Marginal Relevance): Balances relevance with diversity
    """
    print("=" * 60)
    print("Creating Retriever")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Search type: {search_type}")
    print(f"  - Top-k: {k}")
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )
    
    print(f"\n✓ Retriever created successfully!")
    
    return retriever


def create_mmr_retriever(vector_store, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5):
    """
    Create a retriever using Maximal Marginal Relevance (MMR).
    
    MMR balances relevance with diversity to avoid redundant results.
    
    Args:
        vector_store: ChromaDB vector store instance
        k: Number of documents to return
        fetch_k: Number of documents to fetch before MMR reranking
        lambda_mult: Diversity factor (0=max diversity, 1=max relevance)
        
    Returns:
        Configured MMR retriever
    """
    print("=" * 60)
    print("Creating MMR Retriever")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Top-k (final): {k}")
    print(f"  - Fetch-k (initial): {fetch_k}")
    print(f"  - Lambda (diversity): {lambda_mult}")
    
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult
        }
    )
    
    print(f"\n✓ MMR Retriever created successfully!")
    
    return retriever


def test_retriever(retriever, queries: list = None):
    """
    Test the retriever with sample queries.
    
    Args:
        retriever: Configured retriever
        queries: List of test queries (uses defaults if None)
    """
    if queries is None:
        queries = [
            "What is the main message of this book?",
            "How can someone achieve success?",
            "What are the key principles discussed?"
        ]
    
    print("\n" + "=" * 60)
    print("Retriever Test")
    print("=" * 60)
    
    for query in queries:
        print(f"\n{'─' * 60}")
        print(f"Query: \"{query}\"")
        print("─" * 60)
        
        # Retrieve documents
        docs = retriever.invoke(query)
        
        print(f"\nRetrieved {len(docs)} documents:")
        
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get('page', 'N/A')
            if isinstance(page, int):
                page += 1  # Convert 0-indexed to 1-indexed
            
            print(f"\n[Document {i}] Page: {page}")
            
            # Show content preview
            preview = doc.page_content[:250].replace('\n', ' ')
            if len(doc.page_content) > 250:
                preview += "..."
            print(f"Content: {preview}")


def compare_retrieval_methods(vector_store, query: str):
    """
    Compare different retrieval methods on the same query.
    
    Args:
        vector_store: ChromaDB vector store instance
        query: Query to test
    """
    print("\n" + "=" * 60)
    print("Retrieval Method Comparison")
    print("=" * 60)
    print(f"\nQuery: \"{query}\"")
    
    # Method 1: Basic similarity search
    print("\n─── Method 1: Similarity Search (k=4) ───")
    sim_retriever = create_retriever(vector_store, search_type="similarity", k=4)
    sim_docs = sim_retriever.invoke(query)
    
    print("Results:")
    for i, doc in enumerate(sim_docs, 1):
        page = doc.metadata.get('page', 0) + 1
        print(f"  {i}. Page {page}: {doc.page_content[:80]}...")
    
    # Method 2: MMR for diversity
    print("\n─── Method 2: MMR Search (k=4, diversity=0.7) ───")
    mmr_retriever = create_mmr_retriever(vector_store, k=4, lambda_mult=0.3)
    mmr_docs = mmr_retriever.invoke(query)
    
    print("Results:")
    for i, doc in enumerate(mmr_docs, 1):
        page = doc.metadata.get('page', 0) + 1
        print(f"  {i}. Page {page}: {doc.page_content[:80]}...")
    
    # Analysis
    print("\n─── Analysis ───")
    sim_pages = set(doc.metadata.get('page', 0) for doc in sim_docs)
    mmr_pages = set(doc.metadata.get('page', 0) for doc in mmr_docs)
    
    print(f"Similarity search pages: {sorted(p+1 for p in sim_pages)}")
    print(f"MMR search pages: {sorted(p+1 for p in mmr_pages)}")
    print(f"MMR typically shows more diverse page sources")


def demonstrate_retrieval_pipeline(query: str = "What are the key habits of successful people?"):
    """
    Demonstrate the complete retrieval pipeline.
    
    Args:
        query: User query to process
    """
    print("=" * 60)
    print("Complete Retrieval Pipeline Demo")
    print("=" * 60)
    
    # Step 1: Load embeddings model
    print("\n[Step 1] Loading embeddings model...")
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
    
    # Step 2: Load vector store
    print("[Step 2] Loading vector store...")
    if not CHROMA_DIR.exists():
        print("Error: Vector store not found. Run step5_vector_store.py first.")
        return
    
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings_model,
        collection_name=COLLECTION_NAME
    )
    print(f"         Loaded {vector_store._collection.count()} vectors")
    
    # Step 3: Create retriever
    print("[Step 3] Creating retriever...")
    retriever = create_retriever(vector_store, k=4)
    
    # Step 4: Process query
    print(f"\n[Step 4] Processing query...")
    print(f"         Query: \"{query}\"")
    
    print("\n         a) Embedding the query...")
    query_embedding = embeddings_model.embed_query(query)
    print(f"         Generated {len(query_embedding)}-dimensional vector")
    
    print("\n         b) Searching vector store...")
    docs = retriever.invoke(query)
    
    print(f"\n         c) Retrieved {len(docs)} relevant documents")
    
    # Step 5: Show results
    print("\n" + "=" * 60)
    print("Retrieved Context")
    print("=" * 60)
    
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get('page', 0) + 1
        print(f"\n[Document {i}] Source: Page {page}")
        print("-" * 40)
        print(doc.page_content[:400])
        if len(doc.page_content) > 400:
            print("...[truncated]...")
    
    return docs


def main():
    """Main function to demonstrate retriever functionality."""
    
    # Check if vector store exists
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        print("Vector store not found!")
        print("Please run step5_vector_store.py first to create the vector store.")
        return
    
    # Load embeddings and vector store
    print("Loading embeddings model and vector store...\n")
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
    
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings_model,
        collection_name=COLLECTION_NAME
    )
    
    print(f"Loaded vector store with {vector_store._collection.count()} vectors\n")
    
    # Create and test retriever
    retriever = create_retriever(vector_store, k=4)
    test_retriever(retriever)
    
    # Compare methods
    compare_retrieval_methods(
        vector_store, 
        "What steps should I take to achieve my goals?"
    )


if __name__ == "__main__":
    main()

