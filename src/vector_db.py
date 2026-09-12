import faiss
import numpy as np

from src.embedder import load_embedding_model


# Load the embedding model once 
embedding_model = load_embedding_model()


def create_vector_index(embeddings):
    embeddings = np.asarray(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search_index(index,query,chunks,top_k=3,):
    query_embedding = embedding_model.encode([query],normalize_embeddings=True,)

    query_embedding = np.asarray(query_embedding).astype("float32")

    scores, indices = index.search(query_embedding,top_k,)

    results = []

    for score, index_position in zip(scores[0],indices[0],):
        results.append({"score": float(score),"document": chunks[index_position],})

    return results