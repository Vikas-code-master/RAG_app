#!/usr/bin/env python3
"""
Step 10: Simple CLI Application
================================

A complete command-line interface for the RAG application.
This provides a user-friendly way to interact with the book Q&A system.

Features:
---------
- Continuous Q&A loop
- Source citations with page numbers
- Graceful exit handling
- Clear formatting
- Help command
- History tracking

Usage:
------
    python step10_cli_app.py

Commands:
---------
    help     - Show available commands
    sources  - Toggle source display
    clear    - Clear screen
    history  - Show question history
    quit/exit - Exit the application
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate


# Default paths
PROJECT_DIR = Path(__file__).parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "book_collection"


class BookQACLI:
    """
    Command-line interface for Book Q&A using RAG.
    """
    
    COMMANDS = {
        'help': 'Show this help message',
        'sources': 'Toggle source display on/off',
        'clear': 'Clear the screen',
        'history': 'Show question history',
        'stats': 'Show session statistics',
        'quit': 'Exit the application',
        'exit': 'Exit the application',
    }
    
    def __init__(self):
        """Initialize the CLI application."""
        self.show_sources = True
        self.history = []
        self.start_time = datetime.now()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAG components."""
        self._print_header("Initializing RAG System")
        
        try:
            # Check vector store
            if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
                self._print_error(
                    "Vector store not found!\n"
                    "Please run the following first:\n"
                    "  1. python step5_vector_store.py"
                )
                sys.exit(1)
            
            # Embeddings
            print("  Loading embeddings model...", end=" ", flush=True)
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
            print("✓")
            
            # Vector store
            print("  Loading vector store...", end=" ", flush=True)
            self.vector_store = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME
            )
            doc_count = self.vector_store._collection.count()
            print(f"✓ ({doc_count} documents)")
            
            # Retriever
            print("  Creating retriever...", end=" ", flush=True)
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            print("✓")
            
            # LLM
            print("  Loading LLM...", end=" ", flush=True)
            self.llm = Ollama(
                model="llama3.2",
                temperature=0.3,
                num_predict=512
            )
            print("✓")
            
            # Prompt
            self.prompt = self._create_prompt()
            
            print("\n  ✓ System ready!\n")
            
        except Exception as e:
            self._print_error(f"Initialization failed: {e}")
            sys.exit(1)
    
    def _create_prompt(self):
        """Create the RAG prompt."""
        template = """You are a knowledgeable assistant helping users understand a book.

Based on the following excerpts from the book, answer the question.
Be helpful, accurate, and cite specific ideas from the text when possible.
If the excerpts don't contain relevant information, say so honestly.

Book Excerpts:
{context}

Question: {question}

Answer:"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _print_header(self, text: str):
        """Print a formatted header."""
        print("\n" + "═" * 60)
        print(f"  {text}")
        print("═" * 60)
    
    def _print_error(self, text: str):
        """Print an error message."""
        print(f"\n❌ Error: {text}")
    
    def _format_docs(self, docs):
        """Format documents for context."""
        parts = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get('page', 0) + 1
            parts.append(f"[Page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)
    
    def _clear_screen(self):
        """Clear the terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def process_query(self, question: str):
        """
        Process a user query.
        
        Args:
            question: User's question
        """
        print("\n" + "─" * 60)
        
        # Retrieve documents
        print("🔍 Searching for relevant content...", flush=True)
        docs = self.retriever.invoke(question)
        
        # Generate answer
        print("🤔 Generating answer...\n")
        context = self._format_docs(docs)
        formatted_prompt = self.prompt.format(
            context=context,
            question=question
        )
        
        answer = self.llm.invoke(formatted_prompt)
        
        # Display answer
        print("─" * 60)
        print("📚 ANSWER")
        print("─" * 60)
        print(f"\n{answer}\n")
        
        # Display sources if enabled
        if self.show_sources:
            print("─" * 60)
            print("📄 SOURCES")
            print("─" * 60)
            for i, doc in enumerate(docs, 1):
                page = doc.metadata.get('page', 0) + 1
                preview = doc.page_content[:100].replace('\n', ' ')
                print(f"  [{i}] Page {page}: {preview}...")
        
        print("─" * 60)
        
        # Add to history
        self.history.append({
            'question': question,
            'answer': answer[:200] + "..." if len(answer) > 200 else answer,
            'sources': len(docs),
            'time': datetime.now()
        })
    
    def show_help(self):
        """Display help information."""
        print("\n" + "─" * 60)
        print("📖 BOOK Q&A - HELP")
        print("─" * 60)
        print("\nAsk any question about the book, or use these commands:\n")
        
        for cmd, desc in self.COMMANDS.items():
            print(f"  {cmd:12} - {desc}")
        
        print("\nTips:")
        print("  • Ask specific questions for better answers")
        print("  • Use 'sources' to toggle source citations")
        print("  • Press Ctrl+C to exit at any time")
        print("─" * 60)
    
    def show_history(self):
        """Display question history."""
        print("\n" + "─" * 60)
        print("📜 QUESTION HISTORY")
        print("─" * 60)
        
        if not self.history:
            print("\n  No questions asked yet.\n")
        else:
            for i, item in enumerate(self.history, 1):
                time_str = item['time'].strftime("%H:%M:%S")
                print(f"\n  [{i}] {time_str}")
                print(f"      Q: {item['question']}")
                print(f"      A: {item['answer'][:80]}...")
        
        print("─" * 60)
    
    def show_stats(self):
        """Display session statistics."""
        duration = datetime.now() - self.start_time
        minutes = duration.seconds // 60
        seconds = duration.seconds % 60
        
        print("\n" + "─" * 60)
        print("📊 SESSION STATISTICS")
        print("─" * 60)
        print(f"\n  Questions asked: {len(self.history)}")
        print(f"  Session duration: {minutes}m {seconds}s")
        print(f"  Sources enabled: {'Yes' if self.show_sources else 'No'}")
        print(f"  Documents in DB: {self.vector_store._collection.count()}")
        print("─" * 60)
    
    def run(self):
        """Run the main CLI loop."""
        self._clear_screen()
        
        # Welcome message
        print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║              📚 BOOK Q&A with RAG 📚                     ║
║                                                          ║
║   Ask questions about your book and get AI-powered      ║
║   answers with source citations.                         ║
║                                                          ║
║   Type 'help' for commands or 'quit' to exit            ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
        """)
        
        # Main loop
        while True:
            try:
                # Get input
                user_input = input("\n💬 You: ").strip()
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Handle commands
                cmd = user_input.lower()
                
                if cmd in ['quit', 'exit', 'q']:
                    self._goodbye()
                    break
                
                elif cmd == 'help':
                    self.show_help()
                
                elif cmd == 'sources':
                    self.show_sources = not self.show_sources
                    status = "enabled" if self.show_sources else "disabled"
                    print(f"\n  📄 Source display {status}")
                
                elif cmd == 'clear':
                    self._clear_screen()
                
                elif cmd == 'history':
                    self.show_history()
                
                elif cmd == 'stats':
                    self.show_stats()
                
                else:
                    # Process as a question
                    self.process_query(user_input)
                
            except KeyboardInterrupt:
                self._goodbye()
                break
            
            except Exception as e:
                self._print_error(f"An error occurred: {e}")
                print("  Please try again or type 'quit' to exit.")
    
    def _goodbye(self):
        """Display goodbye message."""
        print("\n")
        print("═" * 60)
        print("  👋 Thanks for using Book Q&A!")
        print(f"  📊 You asked {len(self.history)} question(s) this session.")
        print("═" * 60)
        print()


def main():
    """Main entry point."""
    cli = BookQACLI()
    cli.run()


if __name__ == "__main__":
    main()

