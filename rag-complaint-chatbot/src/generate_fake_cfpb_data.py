import pandas as pd
import numpy as np
from pathlib import Path
import random

def generate_fake_complaints(num_complaints: int = 5000) -> pd.DataFrame:
    """Generate fake CFPB complaint data with realistic narratives."""
    
    # Product categories
    products = ['Credit card', 'Credit card or prepaid card', 'Personal loan', 
                'Savings account', 'Money transfer']
    
    # Issues per product
    issues_by_product = {
        'Credit card': ['Billing disputes', 'Late fees', 'Interest rate increases', 
                       'Unauthorized charges', 'Credit limit reduction', 'Payment processing'],
        'Credit card or prepaid card': ['Account opening', 'Funds not available', 
                                       'Card not working', 'Balance disputes'],
        'Personal loan': ['Loan modification', 'Payment processing', 'Interest rate issues',
                          'Collection practices', 'Application denial'],
        'Savings account': ['Account management', 'Deposit issues', 'Withdrawal problems',
                           'Interest calculation', 'Account closure'],
        'Money transfer': ['Transfer not received', 'Transfer delays', 'Wrong recipient',
                          'Fee disputes', 'International transfer issues']
    }
    
    # Companies
    companies = ['CrediTrust Financial', 'Bank of East Africa', 'FinanceFirst Ltd',
                'QuickMoney Corp', 'SecureBank International', 'MetroFinance']
    
    # States
    states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    
    # Realistic complaint narratives templates
    narrative_templates = {
        'Credit card': [
            "I have been a customer for {years} years. Recently I noticed {issue} on my credit card statement. I contacted customer service multiple times but they keep giving me the runaround. This is unacceptable and I want this resolved immediately.",
            "I'm writing to complain about {issue} on my credit card. The charges appeared without my authorization. I've disputed this {times} times but nothing has been done. This is affecting my credit score.",
            "My credit card company suddenly {issue} without proper notification. I've never missed a payment and have excellent credit. This is unfair treatment of a loyal customer.",
            "I'm experiencing {issue} with my credit card. The customer service representatives are unhelpful and keep transferring me between departments. I need this resolved.",
            "There's been {issue} on my account for the past {months} months. Despite my repeated attempts to fix this, the issue persists. I'm considering closing my account."
        ],
        'Credit card or prepaid card': [
            "I applied for a prepaid card but {issue}. It's been {days} days and I still haven't received my card or my money back.",
            "My prepaid card shows {issue} even though I have sufficient funds. This is embarrassing when I try to make purchases.",
            "I loaded money onto my prepaid card but {issue}. The company is not responding to my inquiries.",
            "There's {issue} with my card. I can't access my own money and customer service is not helping.",
            "I've been trying to {issue} but the system keeps rejecting my requests without explanation."
        ],
        'Personal loan': [
            "I took out a personal loan and now {issue}. The terms were clearly explained but the company is not honoring them.",
            "I'm having trouble with {issue} on my personal loan. The interest rate keeps changing without notice.",
            "I applied for a loan modification due to hardship but {issue}. The collection department keeps calling me.",
            "My loan application was denied due to {issue}. I believe this is unfair as I meet all the requirements.",
            "There's {issue} with my loan payments. The company claims I missed payments but I have proof of payment."
        ],
        'Savings account': [
            "I've been trying to {issue} from my savings account but the bank is blocking my transactions without reason.",
            "My savings account shows {issue} even though I deposited the money {days} days ago.",
            "The interest rate on my savings account was {issue} without any notification. This is not what was promised.",
            "I'm having issues with {issue} on my account. The bank charges are excessive and not explained.",
            "I requested to close my savings account but {issue}. It's been {months} months and the account is still active."
        ],
        'Money transfer': [
            "I sent a money transfer {days} days ago but {issue}. The recipient hasn't received anything and the company can't track it.",
            "There's {issue} with my international transfer. The fees were much higher than quoted.",
            "I accidentally sent money to the wrong account and {issue}. The company is refusing to help me recover my funds.",
            "My money transfer was {issue} without explanation. This was an urgent payment and now I'm facing penalties.",
            "I'm experiencing {issue} with regular transfers I make to family. The service has become unreliable."
        ]
    }
    
    # Generate complaints
    complaints = []
    
    for i in range(num_complaints):
        product = random.choice(products)
        issues = issues_by_product.get(product, issues_by_product['Credit card'])
        issue = random.choice(issues)
        company = random.choice(companies)
        state = random.choice(states)
        
        # Generate date (random date in 2023-2024)
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2024-12-31')
        date_received = start_date + (end_date - start_date) * random.random()
        
        # Generate narrative
        templates = narrative_templates.get(product, narrative_templates['Credit card'])
        template = random.choice(templates)
        
        narrative = template.format(
            years=random.randint(1, 10),
            issue=issue.lower(),
            times=random.randint(2, 5),
            months=random.randint(2, 12),
            days=random.randint(3, 30),
            product=product.lower()
        )
        
        # Generate sub-issue
        sub_issues = ['Detailed issue description', 'Specific problem details', 
                     'Additional context provided', 'Follow-up on previous complaint']
        sub_issue = random.choice(sub_issues)
        
        complaint = {
            'Complaint ID': f'CFPB_{random.randint(100000, 999999)}',
            'Date received': date_received.strftime('%Y-%m-%d'),
            'Product': product,
            'Sub-product': f'{product} - {random.choice(["Standard", "Premium", "Basic"])}',
            'Issue': issue,
            'Sub-issue': sub_issue,
            'Consumer complaint narrative': narrative,
            'Company': company,
            'State': state,
            'ZIP code': f'{random.randint(10000, 99999)}',
            'Submitted via': random.choice(['Web', 'Phone', 'Mail', 'Referral']),
            'Company response to consumer': random.choice(['Closed', 'In progress', 'Untimely response']),
            'Timely response?': random.choice(['Yes', 'No']),
            'Consumer disputed?': random.choice(['Yes', 'No'])
        }
        
        complaints.append(complaint)
    
    return pd.DataFrame(complaints)


def main():
    """Generate and save fake CFPB dataset."""
    print("=== Generating Fake CFPB Dataset ===\n")
    
    # Generate fake data
    df = generate_fake_complaints(num_complaints=5000)
    
    # Save to CSV
    output_path = Path('../data/raw/Full CFPB Dataset.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"Generated {len(df)} fake complaints")
    print(f"Saved to: {output_path}")
    print("\nProduct distribution:")
    print(df['Product'].value_counts())
    print("\nSample complaint:")
    print(df.iloc[0]['Consumer complaint narrative'])


if __name__ == "__main__":
    main()
