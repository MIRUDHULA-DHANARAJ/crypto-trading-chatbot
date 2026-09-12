from src.vector_db import search_index


def retrieve_context(index, query, chunks, top_k=3):
    results = search_index(index=index,query=query,chunks=chunks,top_k=top_k,)

    return results