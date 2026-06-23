import gradio as gr
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / 'src'))

from rag_pipeline import RAGPipeline


class ComplaintChatbot:
    """Gradio interface for RAG complaint analysis."""
    
    def __init__(self, vector_store_path: str = '../vector_store'):
        self.pipeline = RAGPipeline(vector_store_path)
        self.conversation_history = []
    
    def process_query(self, question: str, history: list) -> tuple:
        """Process user query and return response with sources."""
        if not question.strip():
            return "", history
        
        # Query the RAG pipeline
        result = self.pipeline.query(question, top_k=5)
        
        # Format response with sources
        answer = result['answer']
        
        # Format sources for display
        sources_text = "\n\n**Sources:**\n"
        for i, doc in enumerate(result['retrieved_docs'][:3], 1):
            metadata = doc['metadata']
            source_info = f"\n{i. **Product:** {metadata.get('product_category', 'N/A')}\n"
            source_info += f"   **Issue:** {metadata.get('issue', 'N/A')}\n"
            source_info += f"   **Text:** {doc['text'][:200]}...\n"
            sources_text += source_info
        
        full_response = answer + sources_text
        
        # Update history
        history.append((question, full_response))
        
        return "", history
    
    def clear_conversation(self) -> tuple:
        """Clear conversation history."""
        self.conversation_history = []
        return [], []


def create_interface():
    """Create Gradio interface."""
    chatbot = ComplaintChatbot()
    
    with gr.Blocks(title="CrediTrust Complaint Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🏦 CrediTrust Financial Complaint Analysis
            
            Ask questions about customer complaints across our financial products:
            - Credit Cards
            - Personal Loans  
            - Savings Accounts
            - Money Transfers
            """
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbox = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_copy_button=True
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What are the main complaints about credit cards?",
                        scale=4
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
            
            with gr.Column(scale=1):
                gr.Markdown("### Example Questions")
                example_questions = [
                    "What are the most common credit card complaints?",
                    "Why are customers unhappy with personal loans?",
                    "What issues appear in money transfer complaints?",
                    "What billing problems are reported?",
                    "How do customers describe fraud issues?",
                    "What are customers saying about fees?"
                ]
                
                for q in example_questions:
                    gr.Button(q, size="sm").click(
                        lambda q=q: q,
                        outputs=question_input
                    )
        
        # Event handlers
        submit_btn.click(
            chatbot.process_query,
            inputs=[question_input, chatbox],
            outputs=[question_input, chatbox]
        )
        
        question_input.submit(
            chatbot.process_query,
            inputs=[question_input, chatbox],
            outputs=[question_input, chatbox]
        )
        
        clear_btn.click(
            chatbot.clear_conversation,
            outputs=[chatbox, question_input]
        )
        
        gr.Markdown(
            """
            ---
            **Note:** This tool uses AI to analyze customer complaints. Always verify critical information with official data sources.
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)