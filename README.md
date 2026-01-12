# RAG Application with LangChain, Ollama, and ChromaDB

A complete Retrieval-Augmented Generation (RAG) application that uses a local LLM to answer questions based on PDF documents.

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  PDF Book   │ ──▶ │  Chunking   │ ──▶ │ Embeddings  │ ──▶ │   Chroma    │
│  (Source)   │     │  (Split)    │     │  (Vectors)  │     │  (Store)    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                                                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Answer    │ ◀── │   Ollama    │ ◀── │   Context   │ ◀── │  Retrieval  │
│  (Output)   │     │   (LLM)     │     │  (Combine)  │     │  (Search)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | LangChain | Orchestration & document processing |
| LLM | Ollama (local) | Text generation |
| Embeddings | nomic-embed-text | Semantic vectors |
| Vector DB | ChromaDB | Similarity search |
| PDF Processing | PyPDF | Document loading |

## Quick Start

### 1. Create Virtual Environment

```bash
cd /home/azureuser/divakar_projects/dl_ai/RAGApp/rag-app

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama Models

```bash
# Install embedding model (required)
ollama pull nomic-embed-text

# Install LLM (choose one)
ollama pull llama3.2          # Recommended for most use cases
ollama pull mistral           # Alternative
ollama pull phi3              # Lighter option
```

### 4. Add Your PDF

Place your PDF file in the `data/` directory (or use the existing PDF in the project root).

### 5. Run the Complete Pipeline

```bash
# Step 1: Verify setup
python step1_setup.py

# Step 2: Load and verify PDF
python step2_load_pdf.py

# Step 3: Test text chunking
python step3_chunking.py

# Step 4: Test embeddings
python step4_embeddings.py

# Step 5: Create vector store (required!)
python step5_vector_store.py

# Step 6: Test retriever
python step6_retriever.py

# Step 7: Test LLM
python step7_llm_setup.py

# Step 8: Test RAG chain
python step8_rag_chain.py

# Step 9: Run queries
python step9_query.py

# Step 10: Run CLI app (interactive)
python step10_cli_app.py
```

## Project Structure

```
rag-app/
├── data/                    # Place PDF files here
│   └── your_book.pdf
├── chroma_db/               # Vector database (auto-created)
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── step1_setup.py           # Dependency verification
├── step2_load_pdf.py        # PDF loading
├── step3_chunking.py        # Text splitting
├── step4_embeddings.py      # Embedding generation
├── step5_vector_store.py    # ChromaDB vector store
├── step6_retriever.py       # Similarity search retriever
├── step7_llm_setup.py       # Ollama LLM configuration
├── step8_rag_chain.py       # RAG chain (retriever + LLM)
├── step9_query.py           # Query execution pipeline
└── step10_cli_app.py        # Interactive CLI application
```

## Step-by-Step Guide

### Step 1: Project Setup

Verifies all dependencies and checks Ollama connectivity:

```bash
python step1_setup.py
```

### Step 2: Load PDF

Load your PDF book using PyPDFLoader:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/your_book.pdf")
documents = loader.load()  # One document per page
```

### Step 3: Text Chunking

Split text into manageable chunks:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
```

**Why chunk?** LLMs have context limits. Smaller chunks enable precise retrieval.

### Step 4: Generate Embeddings

Convert text to semantic vectors:

```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectors = embeddings.embed_documents([c.page_content for c in chunks])
```

### Step 5: Vector Store (Chroma)

Store embeddings in ChromaDB for persistence:

```python
from langchain_chroma import Chroma

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="book_collection"
)
```

**Reload without re-embedding:**
```python
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="book_collection"
)
```

### Step 6: Retriever

Create a retriever for similarity search:

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # Return top 4 matches
)

docs = retriever.invoke("What is the main theme?")
```

### Step 7: LLM Setup

Configure Ollama for text generation:

```python
from langchain_community.llms import Ollama

llm = Ollama(
    model="llama3.2",
    temperature=0.3,  # Lower = more focused
    num_predict=512   # Max tokens
)
```

### Step 8: RAG Chain

Combine retriever + LLM using LCEL:

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

prompt = PromptTemplate(
    template="""Context: {context}
    
Question: {question}

Answer based on the context:""",
    input_variables=["context", "question"]
)

chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
)
```

### Step 9: Query Execution

Execute queries with source tracking:

```python
from step9_query import RAGQueryEngine

engine = RAGQueryEngine()
result = engine.query("What are the key principles?")

print(result.answer)
for source in result.sources:
    print(f"Page {source['page']}: {source['preview']}")
```

### Step 10: CLI Application

Run the interactive CLI:

```bash
python step10_cli_app.py
```

**Commands:**
- `help` - Show available commands
- `sources` - Toggle source display
- `history` - Show question history
- `quit` - Exit the application

## Configuration Options

### Chunking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| chunk_size | 1000 | Max characters per chunk |
| chunk_overlap | 200 | Overlapping characters |

### LLM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| temperature | 0.3 | Randomness (0=deterministic, 1=creative) |
| num_predict | 512 | Max tokens to generate |
| top_p | 0.9 | Nucleus sampling threshold |
| top_k | 40 | Vocabulary limit per token |

### Retriever Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| k | 4 | Number of documents to retrieve |
| search_type | similarity | Search method (similarity/mmr) |

## Troubleshooting

### Ollama Connection Error

```bash
# Make sure Ollama is running
ollama serve

# Check if models are installed
ollama list
```

### Vector Store Not Found

Run step 5 to create the vector store:
```bash
python step5_vector_store.py
```

### Memory Issues

- Reduce `chunk_size`
- Process in batches
- Use a lighter embedding model

### PDF Loading Issues

- Ensure PDF is not password-protected
- Try converting scanned PDFs to searchable PDFs
- Check file permissions

## License

MIT License - Feel free to use and modify.
