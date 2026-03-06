from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids =  []

    # build page content robustly: CSV may use 'Review' or 'Reviews' (or other names)
    text_fields = ["Reviews", "Review", "reviews", "review", "Body", "body", "Text", "text", "Content", "content"]

    for i, row in df.iterrows():
        parts = []
        if "Title" in row and pd.notna(row.get("Title")):
            parts.append(str(row.get("Title")))

        # pick the first available review/text field
        for field in text_fields:
            if field in row and pd.notna(row.get(field)):
                parts.append(str(row.get(field)))
                break

        page_content = " ".join(parts).strip()
        metadata = {}
        if "Rating" in row:
            metadata["rating"] = row.get("Rating")
        if "Date" in row:
            metadata["date"] = row.get("Date")

        if not page_content:
            continue

        document = Document(
            page_content=page_content,
            metadata=metadata,
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

db = Chroma(collection_name="reviews", embedding_function=embeddings, persist_directory=db_location)

vectore_store = Chroma(collection_name="reviews", embedding_function=embeddings, persist_directory=db_location)

if add_documents:
    vectore_store.add_documents(documents, ids=ids)

retriever = vectore_store.as_retriever(
    search_kwargs={"k": 5}

)