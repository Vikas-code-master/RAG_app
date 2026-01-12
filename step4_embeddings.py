"""
Step 4: Generate Embeddings with Ollama
=======================================

This script demonstrates how to convert text chunks into embeddings using
Ollama's nomic-embed-text model.

What are Embeddings?
--------------------
Embeddings are dense vector representations of text that capture semantic meaning.
Similar texts have similar embeddings (close in vector space), enabling:

1. **Semantic Search**: Find documents by meaning, not just keywords
   - "car" and "automobile" would be similar
   - "bank" (financial) and "river bank" would be different

2. **Context Understanding**: Embeddings capture relationships between concepts
   - "king - man + woman ≈ queen" (famous word2vec example)

3. **Efficient Comparison**: Compare thousands of documents quickly using
   vector similarity (cosine similarity, dot product)

nomic-embed-text Model:
-----------------------
- Open-source embedding model optimized for RAG applications
- 768-dimensional embeddings
- Good balance of quality and speed for local inference
- Works well with Ollama for local deployment

How it Works:
-------------
1. Each text chunk is passed through the embedding model
2. The model outputs a fixed-length vector (768 dimensions)
3. This vector numerically represents the semantic content
4. Similar texts produce vectors that are close together
"""

from pathlib import Path
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_ollama_embeddings(model: str = "nomic-embed-text"):
    """
    Create an Ollama embeddings instance.
    
    Args:
        model: Name of the Ollama embedding model
        
    Returns:
        OllamaEmbeddings instance
    """
    embeddings = OllamaEmbeddings(
        model=model,
        # Optional: specify Ollama server URL if not using default
        # base_url="http://localhost:11434"
    )
    return embeddings


def generate_embeddings(chunks: list, embeddings_model) -> list:
    """
    Generate embeddings for a list of text chunks.
    
    Args:
        chunks: List of Document objects or strings
        embeddings_model: The embeddings model to use
        
    Returns:
        List of embedding vectors
    """
    print("=" * 60)
    print("Generating Embeddings")
    print("=" * 60)
    
    # Extract text content from chunks
    if hasattr(chunks[0], 'page_content'):
        texts = [chunk.page_content for chunk in chunks]
    else:
        texts = chunks
    
    print(f"\nProcessing {len(texts)} chunks...")
    print("This may take a moment depending on your hardware.\n")
    
    # Generate embeddings
    embeddings = embeddings_model.embed_documents(texts)
    
    print(f"✓ Generated {len(embeddings)} embeddings")
    print(f"✓ Embedding dimension: {len(embeddings[0])}")
    
    return embeddings


def analyze_embeddings(embeddings: list, chunks: list = None):
    """
    Analyze the generated embeddings and demonstrate similarity.
    
    Args:
        embeddings: List of embedding vectors
        chunks: Optional list of corresponding text chunks
    """
    print("\n" + "=" * 60)
    print("Embedding Analysis")
    print("=" * 60)
    
    # Convert to numpy for analysis
    embeddings_array = np.array(embeddings)
    
    print(f"\nEmbedding Statistics:")
    print(f"  - Number of embeddings: {len(embeddings)}")
    print(f"  - Dimensions: {embeddings_array.shape[1]}")
    print(f"  - Memory size: {embeddings_array.nbytes / 1024:.1f} KB")
    
    # Value statistics
    print(f"\nValue Range (across all embeddings):")
    print(f"  - Min value: {embeddings_array.min():.4f}")
    print(f"  - Max value: {embeddings_array.max():.4f}")
    print(f"  - Mean value: {embeddings_array.mean():.4f}")
    print(f"  - Std deviation: {embeddings_array.std():.4f}")
    
    # Demonstrate similarity calculation
    if len(embeddings) >= 2:
        print(f"\n{'=' * 60}")
        print("Similarity Demonstration")
        print("=" * 60)
        
        # Calculate cosine similarity between first few chunks
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Compare first chunk with others
        print("\nComparing Chunk 1 with other chunks:")
        first_emb = embeddings_array[0]
        
        similarities = []
        for i in range(1, min(6, len(embeddings))):
            sim = cosine_similarity(first_emb, embeddings_array[i])
            similarities.append((i + 1, sim))
            print(f"  - Chunk 1 ↔ Chunk {i + 1}: {sim:.4f}")
        
        # Find most similar
        most_similar = max(similarities, key=lambda x: x[1])
        print(f"\n✓ Most similar to Chunk 1: Chunk {most_similar[0]} "
              f"(similarity: {most_similar[1]:.4f})")


def demonstrate_semantic_search(embeddings_model, sample_texts: list = None):
    """
    Demonstrate how embeddings enable semantic search.
    
    Args:
        embeddings_model: The embeddings model to use
        sample_texts: Optional custom texts to compare
    """
    print("\n" + "=" * 60)
    print("Semantic Search Demonstration")
    print("=" * 60)
    
    if sample_texts is None:
        # Default demonstration texts
        sample_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "AI and ML are transforming the technology industry.",
            "The weather today is sunny and warm.",
            "Deep learning uses neural networks with many layers.",
            "I enjoy eating pizza on Friday nights.",
        ]
    
    print("\nSample texts for comparison:")
    for i, text in enumerate(sample_texts, 1):
        print(f"  {i}. {text}")
    
    # Generate embeddings for samples
    print("\nGenerating embeddings for samples...")
    sample_embeddings = embeddings_model.embed_documents(sample_texts)
    
    # Calculate all pairwise similarities
    def cosine_similarity(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    print("\nSemantic Similarity Matrix:")
    print("   ", end="")
    for i in range(len(sample_texts)):
        print(f"  [{i+1}]  ", end="")
    print()
    
    for i, emb_i in enumerate(sample_embeddings):
        print(f"[{i+1}]", end="")
        for j, emb_j in enumerate(sample_embeddings):
            sim = cosine_similarity(emb_i, emb_j)
            print(f" {sim:.3f}", end="")
        print()
    
    print("\nObservations:")
    print("  - Texts 1, 2, 4 (about AI/ML) have high similarity")
    print("  - Texts 3 and 5 (unrelated topics) have low similarity to AI texts")
    print("  - This enables finding relevant content by meaning, not keywords!")


def main():
    """Main function to demonstrate embeddings generation."""
    
    # Create embeddings model
    print("Initializing Ollama embeddings model...")
    print("Model: nomic-embed-text\n")
    
    try:
        embeddings_model = create_ollama_embeddings()
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running (ollama serve)")
        print("  2. nomic-embed-text is installed (ollama pull nomic-embed-text)")
        return
    
    # Find and load PDF
    data_dir = Path(__file__).parent / "data"
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if pdf_files:
        # Load and chunk the PDF
        pdf_path = str(pdf_files[0])
        print(f"Loading PDF: {pdf_path}\n")
        
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"Loaded {len(documents)} pages → {len(chunks)} chunks\n")
        
        # Generate embeddings (limit to first 10 for demo)
        demo_chunks = chunks[:10]
        embeddings = generate_embeddings(demo_chunks, embeddings_model)
        
        # Analyze embeddings
        analyze_embeddings(embeddings, demo_chunks)
    else:
        print("No PDF found. Running with sample texts instead.\n")
    
    # Always show semantic search demonstration
    demonstrate_semantic_search(embeddings_model)


if __name__ == "__main__":
    main()

