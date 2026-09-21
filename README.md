# RAG Knowledge Assistant

A general-purpose Retrieval-Augmented Generation (RAG) chatbot. Point it at any set of documents — technical docs, notes, manuals, research, whatever — and it answers questions grounded in that content. Not tied to any single domain.

The application supports conversational text interaction and is designed to be extended with voice input/output using the same RAG backend.

## Live Demo

https://crypto-trading-assistant.streamlit.app/

*(demo instance currently loaded with a sample knowledge base — swap `data/raw/` for your own documents to repurpose it)*

## Features

* Retrieval-Augmented Generation (RAG) over any document set you provide
* **Hybrid search** — semantic (FAISS + Sentence Transformers) + keyword (BM25) retrieval combined
* **Cross-encoder reranking** — reranks retrieved chunks for relevance before generation
* **Conversation memory** — recent chat turns are passed to the model so follow-ups work naturally
* **Live PDF updates** — upload a PDF from the app; it's chunked, embedded, and added to the knowledge base instantly, no restart needed
* Source display for every retrieved document
* Grounded answers — handles out-of-domain questions without fabricating facts
* Distinguishes static/reference information from current/live information, where relevant
* Streamlit web interface
* Groq-powered LLM generation
* Designed for future voice input/output

## Architecture

```text
User
 │
 ├── Text Input / PDF Upload
       │
       ▼
   Question
       │
       ▼
  Hybrid Retrieval (FAISS + BM25)
       │
       ▼
   Cross-Encoder Rerank
       │
       ▼
   Top-K Context + Conversation History
       │
       ▼
       LLM
       │
       ├── Grounded Answer + Sources
```

### RAG Pipeline

```text
Knowledge Documents (.md, .txt, or uploaded PDF)
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
   FAISS + BM25 Index
        │
        ▼
  Hybrid Retrieval + Rerank
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
* **Sentence Transformers** — text embeddings + cross-encoder reranking
* **FAISS** — vector similarity search
* **rank_bm25** — keyword search
* **Groq** — LLM inference
* **python-dotenv** — local environment configuration

## Knowledge Base

The assistant works with whatever documents you provide — there's no fixed subject matter. Drop `.md` / `.txt` files into `data/raw/`, or upload PDFs live through the running app.

The system prompt (in `src/generator.py`) can be adjusted per use case — e.g. instructing the model on how to handle domain-specific questions, cite sources, or distinguish reference vs. current information — but the retrieval pipeline itself makes no assumptions about the content's subject.

## Project Structure

```text
rag-knowledge-assistant/
│
├── app/
│   └── main.py
│
├── data/
│   └── raw/
│       └── (your own .md / .txt source documents)
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

### 5. Add your knowledge base

Drop `.md` / `.txt` files into `data/raw/`, or upload PDFs directly from the running app.

### 6. Run the application

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

The retrieval pipeline was evaluated using 20 test questions against the sample knowledge base (18 in-domain, 2 out-of-domain):

| Metric   | Result |
| -------- | -----: |
| Recall@1 |  72.2% |
| Recall@3 |   100% |
| MRR      |  0.861 |

The correct source appeared within the top 3 retrieved results for all 18 in-domain questions. Lower Recall@1 mainly came from semantically related documents outranking the expected source, with the relevant source still retrieved at rank 2. Two negative tests confirmed the system doesn't fabricate answers for out-of-domain or prediction-style queries.
