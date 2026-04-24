import os
from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import sys
import warnings

warnings.filterwarnings("ignore")

load_dotenv(".env")

# ----------------------------
# DOCUMENT LOADER
# ----------------------------
def load_all_docs():
    folder_path = "data/pdfs"
    os.makedirs(folder_path, exist_ok=True)

    all_docs = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            all_docs.extend(loader.load())

    txt_path = "data/python.txt"
    if os.path.exists(txt_path):
        all_docs.extend(TextLoader(txt_path, encoding="utf-8").load())

    print(f"Loaded documents: {len(all_docs)}")
    return all_docs


# ----------------------------
# TEXT SPLITTER
# ----------------------------
def split_docs(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


# ----------------------------
# EMBEDDINGS
# ----------------------------
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Embedding dim:", self.model.get_sentence_embedding_dimension())

    def embed(self, texts):
        return self.model.encode(texts)


# ----------------------------
# VECTOR STORE
# ----------------------------
class VectorStoreManager:
    def __init__(self, path="data/vector_store", name="pdf_documents"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=name)

        print(f"Collection: {name}")
        print(f"Docs in DB: {self.collection.count()}")

    def add_documents(self, docs, embeddings):
        ids = [str(uuid.uuid4()) for _ in docs]

        self.collection.add(
            ids=ids,
            documents=[d.page_content for d in docs],
            metadatas=[d.metadata for d in docs],
            embeddings=[e.tolist() for e in embeddings]
        )

        print("Added:", len(docs))


# ----------------------------
# RETRIEVER
# ----------------------------
class RAGRetriever:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5):
        q_emb = self.embedder.embed([query])[0]

        res = self.vector_store.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=top_k
        )

        docs = []
        if res["documents"]:
            for i in range(len(res["documents"][0])):
                docs.append({
                    "document": res["documents"][0][i],
                    "score": 1 - (res["distances"][0][i] / 2)
                })

        return docs


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():
    load_dotenv()

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "summary"

    print("\n--- Starting Pipeline ---")

    embedder = EmbeddingManager()
    store = VectorStoreManager()
    retriever = RAGRetriever(embedder, store)

    # Ingest only if empty
    if store.collection.count() == 0:
        print("Ingesting documents...")
        docs = load_all_docs()

        if not docs:
            print("No documents found!")
            return

        chunks = split_docs(docs)
        embeddings = embedder.embed([c.page_content for c in chunks])
        store.add_documents(chunks, embeddings)

    print(f"\nQuery: {query}")

    results = retriever.retrieve(query, top_k=5)

    print(f"Retrieved {len(results)} documents\n")

    if len(results) == 0:
        print("❌ No relevant results found")
        return

    for r in results:
        print("Score:", r["score"])
        print(r["document"][:200])
        print("-" * 50)


if __name__ == "__main__":
    main()