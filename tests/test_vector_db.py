from src.loader import load_documents
from src.chunker import chunk_documents
from src.embedder import embed_documents
from src.vector_db import create_vector_index, search_index


documents = load_documents()
chunks = chunk_documents(documents)

embeddings = embed_documents(chunks)

index = create_vector_index(embeddings)

query = "What happens when a leveraged trade loses too much money?"

results = search_index(
    index,
    query,
    chunks,
    top_k=3,
)

print("\nQuery:", query)

for i, result in enumerate(results, start=1):
    print(f"\n--- RESULT {i} ---")
    print("Score:", result["score"])
    print("Source:", result["document"].metadata.get("source"))
    print(result["document"].page_content[:500])