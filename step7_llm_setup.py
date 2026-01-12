"""
Step 7: LLM Setup with Ollama
=============================

This script demonstrates how to configure a local LLM using Ollama for
the generation component of the RAG pipeline.

Why Local LLMs?
---------------
1. **Privacy**: Data stays on your machine
2. **Cost**: No API fees after initial setup
3. **Speed**: No network latency for local inference
4. **Offline**: Works without internet connection
5. **Customization**: Fine-tune for specific use cases

Recommended Models:
-------------------
| Model      | Size  | Speed   | Quality | Best For                |
|------------|-------|---------|---------|-------------------------|
| llama3.2   | 3B    | Fast    | Good    | General purpose         |
| llama3.1   | 8B    | Medium  | Better  | Complex reasoning       |
| mistral    | 7B    | Medium  | Good    | Instruction following   |
| phi3       | 3.8B  | Fast    | Good    | Compact, efficient      |
| gemma2     | 9B    | Medium  | Better  | Balanced performance    |

Key Parameters:
---------------
- temperature: Controls randomness (0=deterministic, 1=creative)
- max_tokens: Maximum length of generated response
- top_p: Nucleus sampling threshold
- top_k: Limits vocabulary for each token
"""

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_llm(
    model: str = "llama3.2",
    temperature: float = 0.3,
    num_predict: int = 512,
    top_p: float = 0.9,
    top_k: int = 40,
):
    """
    Create and configure an Ollama LLM instance.
    
    Args:
        model: Ollama model name
        temperature: Randomness (0-1). Lower = more focused
        num_predict: Max tokens to generate
        top_p: Nucleus sampling (0-1)
        top_k: Top-k sampling
        
    Returns:
        Configured Ollama LLM instance
        
    Temperature Guide:
    - 0.0-0.3: Factual, focused responses (good for RAG)
    - 0.4-0.7: Balanced creativity
    - 0.8-1.0: Creative, varied responses
    """
    print("=" * 60)
    print("LLM Configuration")
    print("=" * 60)
    print(f"\nModel: {model}")
    print(f"Parameters:")
    print(f"  - Temperature: {temperature}")
    print(f"  - Max tokens: {num_predict}")
    print(f"  - Top-p: {top_p}")
    print(f"  - Top-k: {top_k}")
    
    llm = Ollama(
        model=model,
        temperature=temperature,
        num_predict=num_predict,
        top_p=top_p,
        top_k=top_k,
    )
    
    print(f"\n✓ LLM initialized successfully!")
    
    return llm


def test_llm(llm, prompt: str = "What is machine learning in one sentence?"):
    """
    Test the LLM with a simple prompt.
    
    Args:
        llm: Configured LLM instance
        prompt: Test prompt
    """
    print("\n" + "=" * 60)
    print("LLM Test")
    print("=" * 60)
    
    print(f"\nPrompt: \"{prompt}\"")
    print("\nGenerating response...")
    print("-" * 60)
    
    response = llm.invoke(prompt)
    
    print(response)
    print("-" * 60)
    print(f"\n✓ Response generated ({len(response)} characters)")


def test_with_prompt_template(llm):
    """
    Demonstrate using prompt templates with the LLM.
    
    Args:
        llm: Configured LLM instance
    """
    print("\n" + "=" * 60)
    print("Prompt Template Demo")
    print("=" * 60)
    
    # Create a prompt template
    template = """You are a helpful assistant. Answer the question based on the context provided.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create a chain
    chain = prompt | llm | StrOutputParser()
    
    # Test with sample data
    sample_context = """
    Machine learning is a subset of artificial intelligence that enables 
    systems to learn and improve from experience without being explicitly 
    programmed. It focuses on developing algorithms that can access data 
    and use it to learn for themselves.
    """
    
    sample_question = "What is the relationship between machine learning and AI?"
    
    print(f"\nContext: {sample_context.strip()}")
    print(f"\nQuestion: {sample_question}")
    print("\nGenerated Answer:")
    print("-" * 60)
    
    response = chain.invoke({
        "context": sample_context,
        "question": sample_question
    })
    
    print(response)
    print("-" * 60)


def compare_temperatures(llm_model: str = "llama3.2"):
    """
    Compare LLM outputs at different temperature settings.
    
    Args:
        llm_model: Model to test
    """
    print("\n" + "=" * 60)
    print("Temperature Comparison")
    print("=" * 60)
    
    prompt = "Explain what makes a person successful in three key points."
    
    temperatures = [0.0, 0.5, 1.0]
    
    for temp in temperatures:
        print(f"\n{'─' * 60}")
        print(f"Temperature: {temp}")
        print("─" * 60)
        
        llm = Ollama(
            model=llm_model,
            temperature=temp,
            num_predict=200
        )
        
        response = llm.invoke(prompt)
        print(response[:500])
        if len(response) > 500:
            print("...[truncated]...")


def create_rag_prompt_template():
    """
    Create a prompt template optimized for RAG applications.
    
    Returns:
        PromptTemplate configured for RAG
    """
    
    # RAG-specific prompt template
    template = """You are a helpful assistant that answers questions based on the provided context.
    
Instructions:
- Answer the question using ONLY the information from the context below
- If the context doesn't contain enough information, say "I don't have enough information to answer this question"
- Be concise and direct in your response
- Do not make up information not present in the context

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    return prompt


def list_available_models():
    """
    List available Ollama models.
    """
    print("=" * 60)
    print("Available Ollama Models")
    print("=" * 60)
    
    try:
        import ollama
        models = ollama.list()
        
        print("\nInstalled models:")
        for model in models.models:
            size_gb = model.size / (1024**3)
            print(f"  - {model.model:30} ({size_gb:.1f} GB)")
            
    except Exception as e:
        print(f"\nError listing models: {e}")
        print("Make sure Ollama is running: ollama serve")


def main():
    """Main function to demonstrate LLM setup."""
    
    # List available models
    list_available_models()
    
    # Create LLM with recommended settings for RAG
    print("\n")
    llm = create_llm(
        model="llama3.2",      # Change based on what you have installed
        temperature=0.3,       # Low temp for factual responses
        num_predict=512,       # Reasonable length for answers
    )
    
    # Test basic generation
    test_llm(llm)
    
    # Test with prompt template
    test_with_prompt_template(llm)
    
    # Show RAG prompt template
    print("\n" + "=" * 60)
    print("RAG Prompt Template")
    print("=" * 60)
    rag_prompt = create_rag_prompt_template()
    print(f"\n{rag_prompt.template}")


if __name__ == "__main__":
    main()

