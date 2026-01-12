"""
Step 1: Project Setup Verification
==================================

This script verifies that all required dependencies are installed correctly.
Run this after setting up your virtual environment and installing requirements.

Setup Instructions:
-------------------
1. Create virtual environment:
   python -m venv venv

2. Activate it:
   - Linux/Mac: source venv/bin/activate
   - Windows: venv\\Scripts\\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Pull required Ollama models:
   ollama pull nomic-embed-text
   ollama pull llama3.2  (or your preferred LLM)
"""

import sys


def verify_dependencies():
    """Verify all required packages are installed."""
    
    required_packages = [
        ("langchain", "LangChain core"),
        ("langchain_community", "LangChain community integrations"),
        ("langchain_text_splitters", "Text splitting utilities"),
        ("chromadb", "Chroma vector database"),
        ("pypdf", "PDF processing"),
        ("ollama", "Ollama Python client"),
    ]
    
    all_installed = True
    print("=" * 60)
    print("RAG Application - Dependency Verification")
    print("=" * 60)
    
    for package_name, description in required_packages:
        try:
            module = __import__(package_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {description:35} ({package_name} v{version})")
        except ImportError:
            print(f"✗ {description:35} ({package_name}) - NOT INSTALLED")
            all_installed = False
    
    print("=" * 60)
    
    if all_installed:
        print("✓ All dependencies are installed correctly!")
        print("\nNext steps:")
        print("1. Place your PDF file in the 'data/' folder")
        print("2. Run step2_load_pdf.py to load and verify the PDF")
    else:
        print("✗ Some dependencies are missing. Please install them:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    return all_installed


def verify_ollama_connection():
    """Verify Ollama is running and accessible."""
    print("\n" + "=" * 60)
    print("Ollama Connection Verification")
    print("=" * 60)
    
    try:
        import ollama
        
        # Try to list available models
        models = ollama.list()
        model_names = [m.model for m in models.models]
        
        print(f"✓ Ollama is running")
        print(f"✓ Available models: {len(model_names)}")
        
        for name in model_names:
            print(f"  - {name}")
        
        # Check for required models
        has_embedding = any("nomic-embed-text" in name for name in model_names)
        has_llm = any(name for name in model_names if "nomic-embed-text" not in name)
        
        print("=" * 60)
        
        if not has_embedding:
            print("⚠ Warning: nomic-embed-text model not found")
            print("  Run: ollama pull nomic-embed-text")
        
        if not has_llm:
            print("⚠ Warning: No LLM model found")
            print("  Run: ollama pull llama3.2 (or another LLM)")
        
        if has_embedding and has_llm:
            print("✓ All required Ollama models are available!")
            
    except Exception as e:
        print(f"✗ Could not connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")


if __name__ == "__main__":
    verify_dependencies()
    verify_ollama_connection()

