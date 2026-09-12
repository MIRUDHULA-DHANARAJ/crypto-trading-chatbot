from src.loader import load_documents
from src.chunker import chunk_documents


documents = load_documents()
chunks = chunk_documents(documents)

print(f"\nDocuments: {len(documents)}")
print(f"Chunks: {len(chunks)}")

for i, chunk in enumerate(chunks[:5]):
    print("\n--- CHUNK", i + 1, "---")
    print("Source:", chunk.metadata.get("source"))
    print("Characters:", len(chunk.page_content))
    print(chunk.page_content[:300])