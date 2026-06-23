import pandas as pd
import numpy as np
from pathlib import Path
import random
from sentence_transformers import SentenceTransformer


def generate_fake_embeddings(num_chunks: int = 10000) -> pd.DataFrame:
    """Generate fake embeddings data with realistic chunks."""
    
    # Load embedding model to get correct dimension
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding_dim = model.get_sentence_embedding_dimension()
    
    # Product categories
    products = ['Credit card', 'Credit card or prepaid card', 'Personal loan', 
                'Savings account', 'Money transfer']
    
    # Issues
    issues = ['Billing disputes', 'Late fees', 'Interest rate increases', 
              'Unauthorized charges', 'Credit limit reduction', 'Payment processing',
              'Loan modification', 'Transfer delays', 'Account management',
              'Deposit issues', 'Collection practices', 'Fee disputes']
    
    # Companies
    companies = ['CrediTrust Financial', 'Bank of East Africa', 'FinanceFirst Ltd',
                'QuickMoney Corp', 'SecureBank International', 'MetroFinance']
    
    # States
    states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    
    # Sample text chunks (realistic complaint excerpts)
    chunk_texts = [
        "I have been a customer for 5 years. Recently I noticed unauthorized charges on my credit card statement.",
        "The charges appeared without my authorization. I've disputed this 3 times but nothing has been done.",
        "My credit card company suddenly increased my interest rate without proper notification.",
        "I'm experiencing billing disputes with my credit card. The customer service representatives are unhelpful.",
        "There's been late fees on my account for the past 6 months. Despite my repeated attempts to fix this.",
        "I applied for a prepaid card but the funds are not available. It's been 10 days and I still haven't received my card.",
        "My prepaid card shows insufficient funds even though I have sufficient balance in my account.",
        "I loaded money onto my prepaid card but the transaction failed. The company is not responding.",
        "There's an issue with card activation. I can't access my own money and customer service is not helping.",
        "I've been trying to activate my card but the system keeps rejecting my requests without explanation.",
        "I took out a personal loan and now the payment processing is problematic. The terms were clearly explained.",
        "I'm having trouble with interest rate issues on my personal loan. The rate keeps changing without notice.",
        "I applied for a loan modification due to hardship but the request was denied. Collection keeps calling.",
        "My loan application was denied due to credit score issues. I believe this is unfair as I meet requirements.",
        "There's a payment discrepancy with my loan payments. The company claims I missed payments but I have proof.",
        "I've been trying to withdraw from my savings account but the bank is blocking my transactions without reason.",
        "My savings account shows deposit not credited even though I deposited the money 5 days ago.",
        "The interest rate on my savings account was reduced without any notification. This is not what was promised.",
        "I'm having issues with excessive charges on my account. The bank charges are excessive and not explained.",
        "I requested to close my savings account but the request is still pending. It's been 3 months.",
        "I sent a money transfer 7 days ago but the recipient hasn't received anything. The company can't track it.",
        "There's a fee discrepancy with my international transfer. The fees were much higher than quoted.",
        "I accidentally sent money to the wrong account and need a reversal. The company refuses to help recover funds.",
        "My money transfer was delayed without explanation. This was an urgent payment and now I'm facing penalties.",
        "I'm experiencing transfer failure with regular transfers I make to family. The service has become unreliable.",
        "Customer service has been unresponsive to my complaints about the account issues I'm facing.",
        "The company's response time is unacceptable. I've been waiting for weeks for a resolution.",
        "I'm frustrated with the lack of transparency in how my account is being managed.",
        "The communication from the company has been poor throughout this process.",
        "I need this resolved quickly as it's affecting my financial situation significantly."
    ]
    
    # Generate chunks
    chunks_data = []
    
    for i in range(num_chunks):
        complaint_id = f'comp_{random.randint(1000, 9999)}'
        product = random.choice(products)
        issue = random.choice(issues)
        company = random.choice(companies)
        state = random.choice(states)
        
        # Generate random date
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2024-12-31')
        date_received = start_date + (end_date - start_date) * random.random()
        
        # Select random chunk text
        text = random.choice(chunk_texts)
        
        # Generate fake embedding (random values in correct range)
        # Real embeddings are typically in range [-1, 1] for normalized models
        embedding = np.random.uniform(-0.5, 0.5, embedding_dim).astype(np.float32)
        
        chunk_info = {
            'chunk_id': f'chunk_{i}',
            'complaint_id': complaint_id,
            'text': text,
            'embedding': embedding,
            'product_category': product,
            'product': product,
            'issue': issue,
            'sub_issue': f'Detailed {issue.lower()}',
            'company': company,
            'state': state,
            'date_received': date_received.strftime('%Y-%m-%d'),
            'chunk_index': random.randint(0, 5),
            'total_chunks': random.randint(1, 6)
        }
        
        chunks_data.append(chunk_info)
    
    df = pd.DataFrame(chunks_data)
    return df


def main():
    """Generate and save fake embeddings parquet."""
    print("=== Generating Fake Embeddings Data ===\n")
    
    # Generate fake embeddings
    df = generate_fake_embeddings(num_chunks=10000)
    
    # Save to parquet
    output_path = Path('../data/raw/complaint_embeddings.parquet')
    df.to_parquet(output_path, index=False)
    
    print(f"Generated {len(df)} fake chunks with embeddings")
    print(f"Embedding dimension: {len(df.iloc[0]['embedding'])}")
    print(f"Saved to: {output_path}")
    print("\nProduct distribution:")
    print(df['product_category'].value_counts())
    print("\nSample chunk:")
    print(df.iloc[0]['text'])


if __name__ == "__main__":
    main()
