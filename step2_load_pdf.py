"""
Step 2: Load the PDF Document
=============================

This script demonstrates how to load a PDF book using LangChain's PyPDFLoader.
The loader extracts text page-by-page, preserving page metadata.

How PyPDFLoader Works:
---------------------
- Reads the PDF file page by page
- Extracts text content from each page
- Creates a Document object for each page with:
  - page_content: The extracted text
  - metadata: Information like source file, page number
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str) -> list:
    """
    Load a PDF file and extract text page-wise.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects, one per page
    """
    # Verify file exists
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"Loading PDF: {pdf_path}")
    print("-" * 60)
    
    # Initialize the loader
    loader = PyPDFLoader(pdf_path)
    
    # Load and split the PDF (returns one document per page)
    documents = loader.load()
    
    print(f"✓ Successfully loaded {len(documents)} pages")
    
    return documents


def verify_documents(documents: list, show_pages: int = 3):
    """
    Display information about loaded documents for verification.
    
    Args:
        documents: List of Document objects
        show_pages: Number of pages to preview (default: 3)
    """
    print("\n" + "=" * 60)
    print("Document Verification")
    print("=" * 60)
    
    # Overall statistics
    total_chars = sum(len(doc.page_content) for doc in documents)
    avg_chars = total_chars // len(documents) if documents else 0
    
    print(f"\nDocument Statistics:")
    print(f"  - Total pages: {len(documents)}")
    print(f"  - Total characters: {total_chars:,}")
    print(f"  - Average chars/page: {avg_chars:,}")
    
    # Preview first few pages
    print(f"\n{'=' * 60}")
    print(f"Preview of First {min(show_pages, len(documents))} Pages:")
    print("=" * 60)
    
    for i, doc in enumerate(documents[:show_pages]):
        print(f"\n--- Page {i + 1} ---")
        print(f"Metadata: {doc.metadata}")
        
        # Show first 500 characters of content
        content_preview = doc.page_content[:500]
        if len(doc.page_content) > 500:
            content_preview += "..."
        
        print(f"Content Preview:\n{content_preview}")
        print(f"\n[Page length: {len(doc.page_content)} characters]")


def main():
    """Main function to demonstrate PDF loading."""
    
    # Default path - update this to your PDF location
    data_dir = Path(__file__).parent / "data"
    
    # Look for any PDF in the data directory
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("=" * 60)
        print("No PDF files found!")
        print("=" * 60)
        print(f"\nPlease add a PDF file to: {data_dir}")
        print("\nExample usage:")
        print("  1. Copy your book PDF to the 'data/' folder")
        print("  2. Run this script again")
        print("\nAlternatively, specify a path directly:")
        print("  python step2_load_pdf.py /path/to/your/book.pdf")
        return None
    
    # Use the first PDF found
    pdf_path = str(pdf_files[0])
    print(f"Found PDF: {pdf_path}\n")
    
    # Load the PDF
    documents = load_pdf(pdf_path)
    
    # Verify the loaded documents
    verify_documents(documents)
    
    return documents


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Use provided path
        pdf_path = sys.argv[1]
        documents = load_pdf(pdf_path)
        verify_documents(documents)
    else:
        main()

