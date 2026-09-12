# ₿ Crypto Trading Knowledge Assistant

A RAG-based cryptocurrency knowledge chatbot that answers questions about crypto, trading, technical analysis, and risk management using a curated knowledge base.

The application supports conversational text interaction and is designed to be extended with voice input/output using the same RAG backend.

## Live Demo

https://crypto-trading-assistant.streamlit.app/

## Features

* Cryptocurrency knowledge chatbot
* Retrieval-Augmented Generation (RAG)
* FAISS vector search
* Sentence Transformers embeddings
* Grounded answers using a curated knowledge base
* Source display for retrieved documents
* Handles out-of-domain questions without fabricating answers
* Distinguishes static reference prices from live market prices
* Educational and risk-aware responses
* Streamlit web interface
* Groq-powered LLM generation
* Designed for future voice input/output

## Architecture

```text
User
 │
 ├── Text Input
 │
 └── Voice Input
       │
       ▼
   Question
       │
       ▼
  FAISS Retrieval
       │
       ▼
   Top-K Context
       │
       ▼
       LLM
       │
       ├── Text Answer
       │
       └── TTS → Voice Output
```

### RAG Pipeline

```text
Knowledge Documents
        │
        ▼
   Document Loading
        │
        ▼
      Chunking
        │
        ▼
    Embeddings
        │
        ▼
      FAISS
        │
        ▼
   Top-K Retrieval
        │
        ▼
    LLM Generation
        │
        ▼
  Grounded Answer
```

## Tech Stack

* **Python**
* **Streamlit** — web application
* **LangChain** — document loading and text splitting
* **Sentence Transformers** — text embeddings
* **FAISS** — vector similarity search
* **Groq** — LLM inference
* **python-dotenv** — local environment configuration

## Knowledge Base

The chatbot currently contains curated information covering:

* Bitcoin
* Ethereum
* Cryptocurrency fundamentals
* Trading concepts
* Technical analysis
* RSI and other indicators
* Risk management
* Leverage and liquidation
* Position sizing
* Risk-reward concepts
* Reference cryptocurrency prices

The reference price data is static and is explicitly identified as such. The application does not claim to provide live market prices.

## Project Structure

```text
crypto-trading-chatbot/
│
├── app/
│   └── main.py
│
├── data/
│   └── raw/
│       ├── bitcoin_basics.md
│       ├── ethereum_basics.md
│       ├── trading_basics.md
│       ├── technical_analysis.md
│       ├── risk_management.md
│       └── crypto_knowledge_base.txt
│
├── src/
│   ├── chunker.py
│   ├── embedder.py
│   ├── generator.py
│   ├── loader.py
│   ├── retriever.py
│   ├── vector_db.py
│   └── voice.py
│
├── tests/
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_loader.py
│   ├── test_rag.py
│   ├── test_retrieval_eval.py
│   └── test_vector_db.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/MIRUDHULA-DHANARAJ/crypto-trading-chatbot.git
cd crypto-trading-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

The `.env` file is excluded from Git using `.gitignore`.

### 5. Run the application

```bash
streamlit run app/main.py
```

## Deployment

The application can be deployed using Streamlit Community Cloud.

For deployment, configure the Groq API key using Streamlit Secrets:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

The API key should never be committed to the repository.

## Evaluation

The retrieval pipeline was tested using representative cryptocurrency questions covering:

* Bitcoin fundamentals
* Ethereum fundamentals
* Trading concepts
* Technical analysis
* Risk management
* Out-of-domain questions
* Price-related questions


