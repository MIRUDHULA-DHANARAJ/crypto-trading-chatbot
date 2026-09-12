from src.loader import load_documents
from src.chunker import chunk_documents
from src.embedder import embed_documents
from src.vector_db import create_vector_index
from src.retriever import retrieve_context



# Evaluation questions

EVALUATION_QUESTIONS = [
    
    "What is RSI?",
    "What is the current Bitcoin price?",
    "What is Machine Learning?",
    "Will Bitcoin go up tomorrow?"
]


#  Load documents


documents = load_documents()

print(f"\nLoaded documents: {len(documents)}")


# Chunk documents

chunks = chunk_documents(documents)

print(f"Created chunks: {len(chunks)}")


#  Create embeddings

embeddings = embed_documents(chunks)

print(f"Embedding shape: {embeddings.shape}")


#  Create FAISS index

index = create_vector_index(embeddings)

print("FAISS index created.")



#  Run retrieval evaluation

print("\n" + "=" * 70)
print("RETRIEVAL EVALUATION")
print("=" * 70)


for i, question in enumerate(EVALUATION_QUESTIONS, start=1):

    results = retrieve_context(index=index,query=question,chunks=chunks,top_k=3,)

    print(f"\n{i}. {question}")

    for rank, result in enumerate(results, start=1):

        source = result["document"].metadata.get("source")

        source_name = source.split("\\")[-1]

        print(
            f"   Top {rank}: "
            f"{source_name} "
            f"(score={result['score']:.3f})"
        )