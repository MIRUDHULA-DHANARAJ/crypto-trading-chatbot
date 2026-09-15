from src.loader import load_documents
from src.chunker import chunk_documents
from src.embedder import embed_documents
from src.vector_db import create_vector_index
from src.retriever import retrieve_context



# Evaluation dataset

EVALUATION_QUESTIONS = [

    # Bitcoin
    {
        "question": "What is Bitcoin?",
        "expected_source": "bitcoin_basics.md",
    },
    {
        "question": "What is a Satoshi?",
        "expected_source": "bitcoin_basics.md",
    },
    {
        "question": "What is Bitcoin halving?",
        "expected_source": "bitcoin_basics.md",
    },

    # Ethereum
    {
        "question": "What is Ethereum?",
        "expected_source": "ethereum_basics.md",
    },
    {
        "question": "What is a smart contract?",
        "expected_source": "ethereum_basics.md",
    },
    {
        "question": "What is gas on Ethereum?",
        "expected_source": "ethereum_basics.md",
    },

    # Trading
    {
        "question": "What is a limit order?",
        "expected_source": "trading_basics.md",
    },
    {
        "question": "What is the difference between spot and futures trading?",
        "expected_source": "trading_basics.md",
    },
    {
        "question": "What is leverage in crypto trading?",
        "expected_source": "trading_basics.md",
    },

    # Technical analysis
    {
        "question": "What is RSI?",
        "expected_source": "technical_analysis.md",
    },
    {
        "question": "What is MACD?",
        "expected_source": "technical_analysis.md",
    },
    {
        "question": "What is a breakout?",
        "expected_source": "technical_analysis.md",
    },

    # Risk management
    {
        "question": "What is liquidation?",
        "expected_source": "risk_management.md",
    },
    {
        "question": "What are the risks of using leverage?",
        "expected_source": "risk_management.md",
    },
    {
        "question": "What is position sizing?",
        "expected_source": "risk_management.md",
    },
    {
        "question": "What is risk-reward ratio?",
        "expected_source": "risk_management.md",
    },

    # Static knowledge-base data
    {
        "question": "What is the reference Bitcoin price?",
        "expected_source": "crypto_knowledge_base.txt",
    },
    {
        "question": "What cryptocurrencies are included in the knowledge base?",
        "expected_source": "crypto_knowledge_base.txt",
    },

    # Negative / out-of-domain tests
    {
        "question": "What is Machine Learning?",
        "expected_source": None,
    },
    {
        "question": "Will Bitcoin go up tomorrow?",
        "expected_source": None,
    },
]


# Load documents

documents = load_documents()

print(f"\nLoaded documents: {len(documents)}")


# Chunk documents

chunks = chunk_documents(documents)

print(f"Created chunks: {len(chunks)}")


# Create embeddings

embeddings = embed_documents(chunks)

print(f"Embedding shape: {embeddings.shape}")


# Create FAISS index

index = create_vector_index(embeddings)

print("FAISS index created.")


# Retrieval evaluation

TOP_K = 3

print("RETRIEVAL EVALUATION")
print("=" * 70)


recall_at_1_results = []
recall_at_3_results = []
reciprocal_ranks = []

out_of_domain_count = 0


for i, item in enumerate(EVALUATION_QUESTIONS, start=1):

    question = item["question"]
    expected_source = item["expected_source"]

    results = retrieve_context(
        index=index,
        query=question,
        chunks=chunks,
        top_k=TOP_K,
    )

    retrieved_sources = []

    print(f"\n{i}. {question}")

    for rank, result in enumerate(results, start=1):

        source = result["document"].metadata.get(
            "source",
            "Unknown source",
        )

        source_name = source.replace("\\", "/").split("/")[-1]

        retrieved_sources.append(source_name)

        print(f"   Top {rank}: "f"{source_name} "f"(score={result['score']:.3f})")

    # Negative / out-of-domain question

    if expected_source is None:

        out_of_domain_count += 1

        print("   Expected source: None (out-of-domain)")
        continue

    # Recall@1

    recall_at_1 = (expected_source in retrieved_sources[:1])

    recall_at_1_results.append(int(recall_at_1))

    # Recall@3

    recall_at_3 = (expected_source in retrieved_sources[:3])

    recall_at_3_results.append(int(recall_at_3))

    # Reciprocal Rank

    reciprocal_rank = 0

    for rank, source in enumerate(retrieved_sources,start=1,):

        if source == expected_source:

            reciprocal_rank = 1 / rank
            break

    reciprocal_ranks.append(reciprocal_rank)

    print(f"   Expected source: {expected_source}")

    print(f"   Recall@1: "f"{'PASS' if recall_at_1 else 'FAIL'}")

    print(f"   Recall@3: "f"{'PASS' if recall_at_3 else 'FAIL'}")

    print(f"   Reciprocal Rank: "f"{reciprocal_rank:.3f}")


# Calculate metrics

recall_at_1 = (sum(recall_at_1_results)/ len(recall_at_1_results))

recall_at_3 = (sum(recall_at_3_results)/ len(recall_at_3_results))

mrr = (sum(reciprocal_ranks)/ len(reciprocal_ranks))


# Final report


print("FINAL RETRIEVAL METRICS")
print("=" * 70)

print(f"Evaluation questions : "f"{len(EVALUATION_QUESTIONS)}")

print(f"In-domain questions  : "f"{len(recall_at_1_results)}")

print(f"Out-of-domain tests  : "f"{out_of_domain_count}")

print(f"Recall@1             : "f"{recall_at_1:.3f} "f"({recall_at_1 * 100:.1f}%)")

print(f"Recall@3             : "f"{recall_at_3:.3f} "f"({recall_at_3 * 100:.1f}%)")

print(f"MRR                  : {mrr:.3f}")

