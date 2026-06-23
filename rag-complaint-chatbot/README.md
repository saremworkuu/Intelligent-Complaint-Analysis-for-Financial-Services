# CrediTrust Financial Complaint Analysis Chatbot

A Retrieval-Augmented Generation (RAG) powered AI system for analyzing customer complaints across financial products at CrediTrust Financial.

## 📋 Project Overview

This project builds an intelligent complaint-answering chatbot that empowers product, support, and compliance teams to understand customer pain points across four major product categories:
- **Credit Cards**
- **Personal Loans**
- **Savings Accounts**
- **Money Transfers**

### Business Context

CrediTrust Financial is a fast-growing digital finance company serving East African markets through a mobile-first platform. With over 500,000 users across three countries, the company receives thousands of customer complaints monthly through in-app channels, email, and regulatory reporting portals.

### Problem Statement

Internal teams face serious bottlenecks:
- Customer Support is overwhelmed by complaint volume
- Product Managers struggle to identify frequent or critical issues
- Compliance & Risk teams are reactive rather than proactive
- Executives lack visibility into emerging pain points

### Solution

This RAG system enables stakeholders to ask plain-English questions about customer complaints and receive synthesized, evidence-backed answers in seconds rather than spending hours manually reading complaints.

### Key Performance Indicators

1. Reduce time to identify major complaint trends from days to minutes
2. Empower non-technical teams to get answers without data analysts
3. Shift from reactive problem-solving to proactive issue identification

## 📁 Project Structure

```
rag-complaint-chatbot/
├── .github/
│   └── workflows/
│       └── unittests.yml          # CI/CD pipeline for unit tests
├── .vscode/
│   └── settings.json              # VSCode configuration
├── data/
│   ├── raw/                       # Raw CFPB dataset and pre-built embeddings
│   │   ├── Full CFPB Dataset.csv  # Complete complaint dataset
│   │   └── complaint_embeddings.parquet  # Pre-built embeddings
│   └── processed/                 # Filtered and processed data
│       └── filtered_complaints.csv
├── vector_store/                  # ChromaDB vector store (persisted)
├── notebooks/
│   └── notebooks/
│       └── 01_eda_preprocessing.ipynb  # EDA and preprocessing notebook
├── src/
│   ├── chunking_embedding.py      # Task 2: Text chunking & embedding pipeline
│   ├── generate_fake_cfpb_data.py  # Fake data generation for testing
│   ├── generate_fake_embeddings.py  # Fake embeddings generation
│   ├── load_prebuilt_vectorstore.py  # Load pre-built embeddings to ChromaDB
│   └── rag_pipeline.py           # Task 3: RAG retrieval & generation
├── tests/                        # Unit tests
├── app.py                        # Task 4: Gradio web interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/saremworkuu/Intelligent-Complaint-Analysis-for-Financial-Services.git
cd rag-complaint-chatbot
```

2. **Create a virtual environment**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Task 1: Exploratory Data Analysis and Data Preprocessing

Run the Jupyter notebook to perform EDA and create the filtered dataset:

```bash
jupyter notebook notebooks/notebooks/01_eda_preprocessing.ipynb
```

**What it does:**
- Loads the full CFPB complaint dataset
- Analyzes product distribution and narrative lengths
- Filters for 4 target products
- Cleans text narratives
- Saves to `data/processed/filtered_complaints.csv`

### Task 2: Text Chunking, Embedding, and Vector Store Indexing

Create embeddings and vector store from the filtered dataset:

```bash
python src/chunking_embedding.py
```

**What it does:**
- Creates a stratified sample of 10K-15K complaints
- Implements text chunking (RecursiveCharacterTextSplitter, 500 chars, 50 overlap)
- Generates embeddings using all-MiniLM-L6-v2
- Creates ChromaDB vector store with metadata
- Saves to `vector_store/` directory

**Generate Fake Data (for testing):**

```bash
# Generate fake CFPB dataset
python src/generate_fake_cfpb_data.py

# Generate fake embeddings
python src/generate_fake_embeddings.py
```

### Task 3: RAG Pipeline and Evaluation

#### Option A: Use Sample Vector Store (from Task 2)

```bash
python src/rag_pipeline.py
```

#### Option B: Load Pre-built Embeddings

First, load the pre-built embeddings from parquet to ChromaDB:

```bash
python src/load_prebuilt_vectorstore.py
```

Then run the RAG pipeline:

```bash
python src/rag_pipeline.py
```

**What it does:**
- Loads vector store (ChromaDB)
- Implements retriever (semantic search with top-k retrieval)
- Implements generator (LLM-based answer generation)
- Runs evaluation on 8 test questions
- Saves results to `data/processed/evaluation_results.csv`

### Task 4: Interactive Chat Interface

Launch the Gradio web interface:

```bash
python app.py
```

The interface will be available at `http://localhost:7860`

**Features:**
- Natural language query input
- AI-generated answers with source citations
- Example questions for quick testing
- Clear conversation button
- Product and issue metadata display

## 🔧 Key Components

### RAGRetriever
Handles semantic search using ChromaDB and sentence-transformers (all-MiniLM-L6-v2).

### RAGGenerator
Generates answers using Mistral-7B LLM with context-aware prompting.

### ComplaintChunker
Implements text chunking with RecursiveCharacterTextSplitter for optimal retrieval.

### VectorStoreManager
Manages ChromaDB collections with metadata tracking.

### ComplaintChatbot
Gradio-based web interface for user interaction.

## 🛠️ Technologies Used

- **Vector Database**: ChromaDB
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **LLM**: Mistral-7B-Instruct-v0.2
- **Text Processing**: LangChain, RecursiveCharacterTextSplitter
- **Web Interface**: Gradio
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## 📊 Data

### Dataset
- **Source**: Consumer Financial Protection Bureau (CFPB)
- **Total Records**: ~464K complaints (full dataset)
- **Sample Size**: 10K-15K complaints (for development)
- **Products**: Credit Card, Personal Loan, Savings Account, Money Transfer

### Pre-built Vector Store
- **Total Chunks**: ~1.37 million (from 464K complaints)
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters

### Metadata Fields
- complaint_id
- product_category
- product
- issue
- sub_issue
- company
- state
- date_received
- chunk_index
- total_chunks

## 🧪 Testing

Run unit tests:

```bash
pytest tests/ -v
```

## 📝 Evaluation

The system is evaluated using qualitative assessment on representative questions. Results include:

- Generated answers
- Retrieved sources
- Quality scores (1-5)
- Analysis comments

Evaluation results are saved in `data/processed/evaluation_results.csv`.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request)

## 📄 License

Internal project for CrediTrust Financial

## 👥 Team

**Facilitators:**
- Kerod
- Mahbubah
- Feven

## 📅 Key Dates

- Challenge Introduction: June 17, 2026
- Interim Submission: June 21, 2026
- Final Submission: June 23, 2026

## 💬 Support

- Slack channel: #all-week7
- Office Hours: Mon–Fri, 08:00–15:00 UTC

## ⚠️ Notes

- This tool uses AI to analyze customer complaints. Always verify critical information with official data sources.
- The pre-built vector store requires significant disk space (~80MB for embeddings).
- Running the full LLM (Mistral-7B) requires GPU or sufficient RAM for quantization.
