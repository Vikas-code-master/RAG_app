"""
Step 8: RAG Chain - Combining Retriever + LLM
==============================================

This script demonstrates how to create a complete RAG chain using
LangChain Expression Language (LCEL) to combine retrieval with generation.

RAG Chain Flow:
---------------
1. User asks a question
2. Question is embedded and used to search vector store
3. Retrieved documents provide context
4. LLM generates answer grounded in the context
5. Response is returned to user

LCEL (LangChain Expression Language):
-------------------------------------
LCEL provides a declarative way to compose chains using the pipe (|) operator.
Benefits:
- Clean, readable syntax
- Automatic streaming support
- Easy to modify and extend
- Built-in error handling

Chain Components:
-----------------
- Retriever: Fetches relevant context
- Prompt: Formats context + question for LLM
- LLM: Generates the response
- Output Parser: Formats the final output
"""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


# Default paths
PROJECT_DIR = Path(__file__).parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "book_collection"


def format_docs(docs):
    """
    Format retrieved documents into a single context string.
    
    Args:
        docs: List of Document objects
        
    Returns:
        Formatted string with all document contents
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get('page', 0) + 1
        formatted.append(f"[Source {i} - Page {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def create_rag_prompt():
    """
    Create a prompt template for RAG.
    
    Returns:
        Configured PromptTemplate
    """
    template = """You are a helpful assistant that answers questions based on the provided context from a book.

Instructions:
- Answer the question using ONLY the information from the context below
- If the context doesn't contain enough information, clearly state that
- Be concise and direct
- Reference the source pages when relevant
- Do not make up information not present in the context

Context:
{context}

Question: {question}

Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def create_rag_chain(retriever, llm):
    """
    Create a RAG chain using LCEL.
    
    Args:
        retriever: Document retriever
        llm: Language model
        
    Returns:
        Configured LCEL chain
    """
    print("=" * 60)
    print("Creating RAG Chain")
    print("=" * 60)
    
    # Get the prompt
    prompt = create_rag_prompt()
    
    # Build the chain using LCEL
    # The chain works as follows:
    # 1. RunnableParallel runs retriever and passes question through
    # 2. format_docs converts retrieved docs to string
    # 3. prompt formats the context and question
    # 4. llm generates the response
    # 5. StrOutputParser extracts the text
    
    chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough()
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("\n✓ RAG chain created!")
    print("\nChain flow:")
    print("  Question → Retriever → Format Docs → Prompt → LLM → Output")
    
    return chain


def create_rag_chain_with_sources(retriever, llm):
    """
    Create a RAG chain that also returns source documents.
    
    Args:
        retriever: Document retriever
        llm: Language model
        
    Returns:
        Chain that returns both answer and sources
    """
    print("=" * 60)
    print("Creating RAG Chain with Sources")
    print("=" * 60)
    
    prompt = create_rag_prompt()
    
    def create_response_with_sources(inputs):
        """Generate response and include source information."""
        docs = inputs["docs"]
        question = inputs["question"]
        
        # Format context
        context = format_docs(docs)
        
        # Generate answer
        formatted_prompt = prompt.format(context=context, question=question)
        answer = llm.invoke(formatted_prompt)
        
        # Extract source info
        sources = []
        for doc in docs:
            sources.append({
                "page": doc.metadata.get('page', 0) + 1,
                "content_preview": doc.page_content[:150] + "..."
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "question": question
        }
    
    # Chain that preserves documents for source tracking
    chain = (
        RunnableParallel(
            docs=retriever,
            question=RunnablePassthrough()
        )
        | create_response_with_sources
    )
    
    print("\n✓ RAG chain with sources created!")
    
    return chain


def test_rag_chain(chain, question: str):
    """
    Test the RAG chain with a question.
    
    Args:
        chain: RAG chain
        question: Question to ask
    """
    print("\n" + "=" * 60)
    print("RAG Chain Test")
    print("=" * 60)
    
    print(f"\nQuestion: \"{question}\"")
    print("\nGenerating answer...")
    print("-" * 60)
    
    response = chain.invoke(question)
    
    if isinstance(response, dict):
        # Response with sources
        print("\nAnswer:")
        print(response["answer"])
        print("\n" + "-" * 60)
        print("\nSources used:")
        for i, source in enumerate(response["sources"], 1):
            print(f"  {i}. Page {source['page']}: {source['content_preview'][:60]}...")
    else:
        # Simple response
        print(response)
    
    print("-" * 60)


def load_components():
    """
    Load all necessary components for the RAG chain.
    
    Returns:
        Tuple of (vector_store, retriever, llm)
    """
    # Check if vector store exists
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise FileNotFoundError(
            "Vector store not found! Please run step5_vector_store.py first."
        )
    
    print("Loading components...")
    
    # Embeddings
    print("  - Loading embeddings model...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Vector store
    print("  - Loading vector store...")
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    print(f"    Found {vector_store._collection.count()} vectors")
    
    # Retriever
    print("  - Creating retriever (k=4)...")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # LLM
    print("  - Loading LLM (llama3.2)...")
    llm = Ollama(
        model="llama3.2",
        temperature=0.3,
        num_predict=512
    )
    
    print("\n✓ All components loaded!\n")
    
    return vector_store, retriever, llm


def main():
    """Main function to demonstrate RAG chain."""
    
    try:
        # Load components
        vector_store, retriever, llm = load_components()
        
        # Create basic RAG chain
        print("\n--- Basic RAG Chain ---")
        basic_chain = create_rag_chain(retriever, llm)
        
        # Test basic chain
        test_rag_chain(
            basic_chain,
            "What are the main principles for success discussed in this book?"
        )
        
        # Create chain with sources
        print("\n\n--- RAG Chain with Sources ---")
        sources_chain = create_rag_chain_with_sources(retriever, llm)
        
        # Test with sources
        test_rag_chain(
            sources_chain,
            "How can I develop a positive mindset?"
        )
        
        return basic_chain, sources_chain
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None


if __name__ == "__main__":
    main()

