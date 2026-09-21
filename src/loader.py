import tempfile
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_documents():
    documents = []

    md_loader = DirectoryLoader(
        str(DATA_PATH),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    txt_loader = DirectoryLoader(
        str(DATA_PATH),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents.extend(md_loader.load())
    documents.extend(txt_loader.load())

    return documents


def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    return documents