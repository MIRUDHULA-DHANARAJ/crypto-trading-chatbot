from src.loader import load_documents


documents = load_documents()

print(f"\nLoaded documents: {len(documents)}")

for document in documents:
    print("\n---")
    print("Source:", document.metadata.get("source"))
    print("Characters:", len(document.page_content))
    print("Preview:", document.page_content[:150])