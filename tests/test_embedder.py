from src.loader import load_documents
from src.chunker import chunk_documents
from src.embedder import embed_documents


documents = load_documents()
chunks = chunk_documents(documents)

embeddings = embed_documents(chunks)

print("\nChunks:", len(chunks))
print("Embedding shape:", embeddings.shape)