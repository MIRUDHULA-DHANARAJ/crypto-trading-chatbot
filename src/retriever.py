import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.vector_db import search_index


# Cross-Encoder
# Do NOT load the model during application startup.
# It will be loaded only when reranking is first requested.

cross_encoder = None


def get_cross_encoder():
    global cross_encoder
    if cross_encoder is None:
        print("Loading CrossEncoder...")
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("CrossEncoder loaded.")
    return cross_encoder


# BM25

def build_bm25_index(chunks):
    tokenized_corpus = []
    for chunk in chunks:
        tokenized_corpus.append(chunk.page_content.split())
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


# Semantic Search
def semantic_search(query, index, chunks, top_k=5):
    results = search_index(index, query, chunks, top_k=top_k)
    return [result["document"] for result in results]


# Keyword Search
def keyword_search(query, bm25, chunks, top_k=5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [chunks[i] for i in top_indices]
    return results


# Hybrid Search
def hybrid_search(query, index, bm25, chunks, top_k=5):
    semantic_results = semantic_search(query, index, chunks, top_k=top_k)
    keyword_results = keyword_search(query, bm25, chunks, top_k=top_k)

    combined_results = []
    seen = set()

    for chunk in (semantic_results + keyword_results):
        key = (chunk.metadata.get("source"), chunk.page_content)
        if key not in seen:
            combined_results.append(chunk)
            seen.add(key)

    return combined_results


# Cross-Encoder Reranking
def rerank(query, candidates, top_k=5):
    encoder = get_cross_encoder()

    pairs = [[query, chunk.page_content] for chunk in candidates]
    scores = encoder.predict(pairs)

    scored_candidates = list(zip(candidates, scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    reranked = [(chunk, float(score)) for chunk, score in scored_candidates[:top_k]]
    return reranked


# Retrieve Context (used by the app)
def retrieve_context(index, bm25, query, chunks, top_k=3):
    candidates = hybrid_search(query, index, bm25, chunks, top_k=10)
    reranked = rerank(query, candidates, top_k=top_k)

    results = []
    for chunk, score in reranked:
        results.append({"score": score, "document": chunk})

    return results