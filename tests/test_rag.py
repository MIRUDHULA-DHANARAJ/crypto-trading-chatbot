from src.loader import load_documents
from src.chunker import chunk_documents
from src.embedder import embed_documents
from src.vector_db import create_vector_index
from src.retriever import retrieve_context
from src.generator import generate_answer


# 1. Load knowledge base
documents = load_documents()

# 2. Split documents into chunks
chunks = chunk_documents(documents)

# 3. Create embeddings
embeddings = embed_documents(chunks)

# 4. Build FAISS index
index = create_vector_index(embeddings)

# 5. Ask a question
query = "What happens when a leveraged trade loses too much money?"

# 6. Retrieve relevant knowledge
results = retrieve_context(
    index=index,
    query=query,
    chunks=chunks,
    top_k=3,
)

# 7. Generate answer using retrieved knowledge
answer = generate_answer(
    query=query,
    retrieved_results=results,
)

print("\n==============================")
print("QUESTION")
print("==============================")
print(query)

print("\n==============================")
print("ANSWER")
print("==============================")
print(answer)

print("\n==============================")
print("SOURCES")
print("==============================")

for result in results:
    source = result["document"].metadata.get("source")
    score = result["score"]

    print(f"- {source} | score: {score:.3f}")