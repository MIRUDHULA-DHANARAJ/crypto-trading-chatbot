from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    model = SentenceTransformer(MODEL_NAME)
    return model


def embed_documents(documents):
    model = load_embedding_model()

    texts = [document.page_content for document in documents]

    embeddings = model.encode(texts,normalize_embeddings=True,show_progress_bar=True,)

    return embeddings