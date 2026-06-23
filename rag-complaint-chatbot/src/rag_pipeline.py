import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class RAGRetriever:
    """Handle retrieval from vector store."""
    
    def __init__(self, vector_store_path: str, collection_name: str = "complaint_chunks"):
        self.client = chromadb.PersistentClient(
            path=vector_store_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(collection_name)
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k relevant chunks for a query."""
        # Embed query
        query_embedding = self.embedding_model.encode(query)
        
        # Search vector store
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            retrieved_docs.append({
                'chunk_id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return retrieved_docs


class RAGGenerator:
    """Handle answer generation using LLM."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        self.prompt_template = self._build_prompt_template()
    
    def _build_prompt_template(self) -> str:
        """Build the prompt template for the LLM."""
        template = """You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints based on the provided context.

Instructions:
- Use ONLY the retrieved complaint excerpts to formulate your answer
- If the context doesn't contain enough information to answer the question, state that clearly
- Be concise and specific
- Focus on actionable insights when possible
- Cite the product categories mentioned in the context

Context:
{context}

Question:
{question}

Answer:"""
        return template
    
    def generate(self, query: str, retrieved_docs: List[Dict], max_new_tokens: int = 512) -> str:
        """Generate answer based on query and retrieved documents."""
        # Combine retrieved chunks into context
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Source {i}] {doc['text']}")
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = self.prompt_template.format(
            context=context,
            question=query
        )
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        answer = response.split("Answer:")[-1].strip()
        
        return answer


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation."""
    
    def __init__(self, vector_store_path: str):
        self.retriever = RAGRetriever(vector_store_path)
        self.generator = RAGGenerator()
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """Process a question through the RAG pipeline."""
        # Retrieve relevant chunks
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
        
        # Generate answer
        answer = self.generator.generate(question, retrieved_docs)
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs
        }


def evaluate_rag_pipeline(pipeline: RAGPipeline, test_questions: List[str]) -> pd.DataFrame:
    """Evaluate RAG pipeline on test questions."""
    results = []
    
    for question in test_questions:
        print(f"\nProcessing: {question}")
        result = pipeline.query(question)
        
        results.append({
            'Question': question,
            'Answer': result['answer'],
            'Sources': [doc['text'][:200] + '...' for doc in result['retrieved_docs'][:2]],
            'Quality Score': '',  # To be filled manually
            'Comments': ''  # To be filled manually
        })
    
    return pd.DataFrame(results)


def main():
    """Main function to run RAG pipeline and evaluation."""
    print("=== Task 3: RAG Pipeline and Evaluation ===\n")
    
    # Initialize pipeline with pre-built vector store
    # Note: You'll need to load the pre-built embeddings from complaint_embeddings.parquet
    # This is a placeholder - you'll need to adapt this to load from the parquet file
    
    # For now, using the sample vector store created in Task 2
    vector_store_path = '../vector_store'
    pipeline = RAGPipeline(vector_store_path)
    
    # Test questions for evaluation
    test_questions = [
        "What are the most common complaints about credit cards?",
        "Why are customers unhappy with personal loans?",
        "What issues do people report with money transfers?",
        "What are the main problems with savings accounts?",
        "What billing issues are mentioned across all products?",
        "How do customers describe customer service problems?",
        "What fraud-related complaints appear in the data?",
        "What are customers saying about fees and charges?"
    ]
    
    # Run evaluation
    print("--- Running Evaluation ---")
    evaluation_df = evaluate_rag_pipeline(pipeline, test_questions)
    
    # Save evaluation results
    output_path = Path('../data/processed/evaluation_results.csv')
    evaluation_df.to_csv(output_path, index=False)
    print(f"\nEvaluation results saved to: {output_path}")
    
    # Display results
    print("\n=== Evaluation Results ===")
    pd.set_option('display.max_colwidth', 100)
    print(evaluation_df[['Question', 'Answer']].to_string())


if __name__ == "__main__":
    main()