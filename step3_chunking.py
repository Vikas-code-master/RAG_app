"""
Step 3: Text Chunking
=====================

This script demonstrates how to split extracted text into manageable chunks
using LangChain's RecursiveCharacterTextSplitter.

Why Chunking is Needed:
-----------------------
1. **Context Window Limits**: LLMs have token limits. Large documents must be
   split into smaller pieces that fit within these limits.

2. **Semantic Relevance**: Smaller chunks allow the retrieval system to find
   the most relevant pieces of information, rather than returning entire pages
   that may contain mostly irrelevant content.

3. **Embedding Quality**: Embedding models work better with focused, coherent
   text segments. Very long texts dilute the semantic meaning in the embedding.

4. **Memory Efficiency**: Processing smaller chunks requires less memory and
   allows for better caching strategies.

RecursiveCharacterTextSplitter:
-------------------------------
This splitter tries to keep semantically related text together by:
1. First trying to split on paragraph breaks (\\n\\n)
2. Then on single newlines (\\n)
3. Then on spaces
4. Finally on characters if needed

This hierarchical approach preserves context better than fixed-size splits.
"""

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def create_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Create a text splitter with specified parameters.
    
    Args:
        chunk_size: Maximum size of each chunk (in characters)
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        Configured RecursiveCharacterTextSplitter
        
    Why these defaults?
    - chunk_size=1000: Provides enough context for meaningful retrieval
      while staying well under token limits
    - chunk_overlap=200: Ensures context isn' t lost at chunk boundaries
      and helps with questions that span chunk edges
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Default separators - split on these in order of preference
        separators=[
            "\n\n",   # Paragraph breaks
            "\n",     # Line breaks
            ". ",     # Sentences
            " ",      # Words
            "",       # Characters (last resort)
        ],
        is_separator_regex=False,
    )
    
    return text_splitter


def chunk_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into smaller chunks.
    
    Args:
        documents: List of Document objects from PDF loader
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of chunked Document objects
    """
    print("=" * 60)
    print("Text Chunking")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Chunk size: {chunk_size} characters")
    print(f"  - Chunk overlap: {chunk_overlap} characters")
    print(f"  - Input documents: {len(documents)}")
    
    # Create the splitter
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    
    # Split the documents
    chunks = text_splitter.split_documents(documents)
    
    print(f"  - Output chunks: {len(chunks)}")
    print(f"  - Avg chunks per page: {len(chunks) / len(documents):.1f}")
    
    return chunks


def analyze_chunks(chunks: list, show_samples: int = 3):
    """
    Analyze and display information about the chunks.
    
    Args:
        chunks: List of chunked Document objects
        show_samples: Number of sample chunks to display
    """
    print("\n" + "=" * 60)
    print("Chunk Analysis")
    print("=" * 60)
    
    # Calculate statistics
    chunk_lengths = [len(chunk.page_content) for chunk in chunks]
    
    print(f"\nChunk Statistics:")
    print(f"  - Total chunks: {len(chunks)}")
    print(f"  - Min chunk size: {min(chunk_lengths)} chars")
    print(f"  - Max chunk size: {max(chunk_lengths)} chars")
    print(f"  - Avg chunk size: {sum(chunk_lengths) // len(chunk_lengths)} chars")
    
    # Distribution
    small = sum(1 for l in chunk_lengths if l < 500)
    medium = sum(1 for l in chunk_lengths if 500 <= l < 900)
    large = sum(1 for l in chunk_lengths if l >= 900)
    
    print(f"\nSize Distribution:")
    print(f"  - Small (<500 chars): {small} ({100*small//len(chunks)}%)")
    print(f"  - Medium (500-900 chars): {medium} ({100*medium//len(chunks)}%)")
    print(f"  - Large (>=900 chars): {large} ({100*large//len(chunks)}%)")
    
    # Show sample chunks
    print(f"\n{'=' * 60}")
    print(f"Sample Chunks (showing {min(show_samples, len(chunks))} of {len(chunks)}):")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks[:show_samples]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Source: Page {chunk.metadata.get('page', 'unknown') + 1}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"\nContent:")
        print("-" * 40)
        # Show first 400 characters
        preview = chunk.page_content[:400]
        if len(chunk.page_content) > 400:
            preview += "\n...[truncated]..."
        print(preview)


def main():
    """Main function to demonstrate text chunking."""
    
    # Find PDF files
    data_dir = Path(__file__).parent / "data"
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in data/ directory.")
        print("Please add a PDF file and try again.")
        return None
    
    # Load the PDF
    pdf_path = str(pdf_files[0])
    print(f"Loading PDF: {pdf_path}\n")
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages\n")
    
    # Chunk the documents
    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
    
    # Analyze the chunks
    analyze_chunks(chunks)
    
    return chunks


if __name__ == "__main__":
    main()

