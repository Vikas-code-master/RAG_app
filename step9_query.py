"""
Step 9: Query Execution
=======================

This script demonstrates the complete query execution pipeline, showing
how a user question flows through retrieval to final answer generation.

Query Pipeline:
---------------
1. Accept user question
2. Retrieve relevant chunks from vector store
3. Format context from retrieved documents
4. Generate answer using LLM
5. Return answer with source citations

This module provides both a function-based API and a class-based API
for different use cases.
"""

from pathlib import Path
from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


# Default paths
PROJECT_DIR = Path(__file__).parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "book_collection"


@dataclass
class QueryResult:
    """Container for query results."""
    question: str
    answer: str
    sources: list
    num_sources: int
    
    def __str__(self):
        result = f"Question: {self.question}\n\n"
        result += f"Answer:\n{self.answer}\n\n"
        result += f"Sources ({self.num_sources} documents):\n"
        for i, source in enumerate(self.sources, 1):
            result += f"  {i}. Page {source['page']}: {source['preview'][:60]}...\n"
        return result


class RAGQueryEngine:
    """
    Complete RAG query engine for question answering over documents.
    """
    
    def __init__(
        self,
        chroma_dir: str = None,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2",
        temperature: float = 0.3,
        top_k: int = 4
    ):
        """
        Initialize the RAG query engine.
        
        Args:
            chroma_dir: Path to ChromaDB directory
            collection_name: Name of the collection
            embedding_model: Ollama embedding model name
            llm_model: Ollama LLM model name
            temperature: LLM temperature (0-1)
            top_k: Number of documents to retrieve
        """
        self.chroma_dir = chroma_dir or str(CHROMA_DIR)
        self.collection_name = collection_name
        self.top_k = top_k
        
        print("=" * 60)
        print("Initializing RAG Query Engine")
        print("=" * 60)
        
        # Initialize embeddings
        print(f"\nLoading embedding model: {embedding_model}")
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Load vector store
        print(f"Loading vector store from: {self.chroma_dir}")
        self.vector_store = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        self.doc_count = self.vector_store._collection.count()
        print(f"  - Loaded {self.doc_count} document vectors")
        
        # Create retriever
        print(f"Creating retriever (top-k={top_k})")
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        
        # Initialize LLM
        print(f"Loading LLM: {llm_model} (temp={temperature})")
        self.llm = Ollama(
            model=llm_model,
            temperature=temperature,
            num_predict=512
        )
        
        # Create prompt template
        self.prompt = self._create_prompt()
        
        print("\n✓ Query engine initialized!")
        print("=" * 60)
    
    def _create_prompt(self):
        """Create the RAG prompt template."""
        template = """You are a helpful assistant answering questions about a book.

Use ONLY the following context to answer. If the context doesn't contain 
the answer, say "I couldn't find information about that in the book."

Context:
{context}

Question: {question}

Provide a clear, helpful answer based on the context above:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _format_docs(self, docs):
        """Format documents into context string."""
        parts = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get('page', 0) + 1
            parts.append(f"[Page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)
    
    def query(self, question: str, verbose: bool = True) -> QueryResult:
        """
        Execute a query and return results.
        
        Args:
            question: User's question
            verbose: Whether to print progress
            
        Returns:
            QueryResult with answer and sources
        """
        if verbose:
            print("\n" + "─" * 60)
            print(f"Query: {question}")
            print("─" * 60)
        
        # Step 1: Retrieve relevant documents
        if verbose:
            print("\n[1/3] Retrieving relevant documents...")
        
        docs = self.retriever.invoke(question)
        
        if verbose:
            print(f"      Retrieved {len(docs)} documents")
        
        # Step 2: Format context
        if verbose:
            print("[2/3] Formatting context...")
        
        context = self._format_docs(docs)
        
        # Step 3: Generate answer
        if verbose:
            print("[3/3] Generating answer...")
        
        formatted_prompt = self.prompt.format(
            context=context,
            question=question
        )
        
        answer = self.llm.invoke(formatted_prompt)
        
        # Prepare sources
        sources = []
        for doc in docs:
            sources.append({
                "page": doc.metadata.get('page', 0) + 1,
                "preview": doc.page_content[:200],
                "full_content": doc.page_content
            })
        
        result = QueryResult(
            question=question,
            answer=answer,
            sources=sources,
            num_sources=len(sources)
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("ANSWER")
            print("=" * 60)
            print(f"\n{answer}")
            print("\n" + "-" * 60)
            print("SOURCES USED")
            print("-" * 60)
            for i, source in enumerate(sources, 1):
                print(f"\n[{i}] Page {source['page']}:")
                preview = source['preview'].replace('\n', ' ')[:150]
                print(f"    {preview}...")
        
        return result
    
    def query_with_streaming(self, question: str):
        """
        Execute a query with streaming output.
        
        Args:
            question: User's question
            
        Yields:
            Tokens as they are generated
        """
        print("\n" + "─" * 60)
        print(f"Query: {question}")
        print("─" * 60)
        
        # Retrieve documents
        print("\nRetrieving context...")
        docs = self.retriever.invoke(question)
        context = self._format_docs(docs)
        
        # Format prompt
        formatted_prompt = self.prompt.format(
            context=context,
            question=question
        )
        
        # Stream response
        print("\nAnswer (streaming):")
        print("-" * 40)
        
        for chunk in self.llm.stream(formatted_prompt):
            print(chunk, end="", flush=True)
            yield chunk
        
        print("\n" + "-" * 40)
        
        # Print sources
        print("\nSources:")
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get('page', 0) + 1
            print(f"  {i}. Page {page}")


def execute_query(question: str, verbose: bool = True) -> QueryResult:
    """
    Convenience function to execute a single query.
    
    Args:
        question: User's question
        verbose: Whether to print progress
        
    Returns:
        QueryResult with answer and sources
    """
    engine = RAGQueryEngine()
    return engine.query(question, verbose=verbose)


def demo_multiple_queries():
    """Demonstrate querying with multiple questions."""
    
    # Initialize engine once
    engine = RAGQueryEngine()
    
    # Sample questions
    questions = [
        "What is the main theme of this book?",
        "What advice does the author give for achieving goals?",
        "What role does persistence play according to the book?",
    ]
    
    print("\n" + "=" * 60)
    print("Multiple Query Demo")
    print("=" * 60)
    
    results = []
    for q in questions:
        result = engine.query(q, verbose=True)
        results.append(result)
        print("\n")
    
    return results


def main():
    """Main function to demonstrate query execution."""
    
    # Check if vector store exists
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        print("Error: Vector store not found!")
        print("Please run step5_vector_store.py first.")
        return
    
    # Create query engine
    engine = RAGQueryEngine()
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Query Mode")
    print("=" * 60)
    print("Enter your questions (type 'quit' to exit)\n")
    
    while True:
        try:
            question = input("\n📖 Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            # Execute query
            result = engine.query(question)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break


if __name__ == "__main__":
    main()

