import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Project imports

from src.loader import load_documents
from src.chunker import chunk_documents
from src.embedder import embed_documents
from src.vector_db import create_vector_index
from src.retriever import retrieve_context
from src.generator import generate_answer


# Page configuration

st.set_page_config(page_title="Crypto Trading Assistant",page_icon="₿",layout="centered",)


# Build knowledge base

@st.cache_resource
def build_knowledge_base():

    documents = load_documents()

    chunks = chunk_documents(documents)

    embeddings = embed_documents(chunks)

    index = create_vector_index(embeddings)

    return index, chunks



# Header

st.title("₿ Crypto Trading Knowledge Assistant")

st.caption("Ask questions about cryptocurrency, trading, ""technical analysis, and risk management.")

st.info("Educational use only. This assistant does not provide ""personalized financial advice or guaranteed predictions.")


# Load knowledge base

with st.spinner("Loading crypto knowledge base..."):

    index, chunks = build_knowledge_base()


# Chat history

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# User input

query = st.chat_input("Ask a question about crypto trading...")


# Process question

if query:

    # Display user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):

        st.markdown(query)


    # Generate assistant response
    

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            # Retrieve relevant chunks
            results = retrieve_context(index=index,query=query,chunks=chunks,top_k=3,)

            # Generate grounded answer
            answer = generate_answer(query=query,retrieved_results=results,)


        # Display answer

        st.markdown(answer)


        # Display sources

        with st.expander("📚 Sources"):

            seen_sources = set()

            for result in results:

                document = result["document"]

                source = document.metadata.get("source","Unknown source",)

                # Works with both Windows and Unix paths
                source_name = (source.replace("\\", "/").split("/")[-1])

                if source_name not in seen_sources:

                    st.markdown(
                        f"- **{source_name}** "
                        f"(similarity: {result['score']:.3f})"
                    )

                    seen_sources.add(source_name)


    # Save assistant response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )