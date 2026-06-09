from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
import json


TRACKING_FILE = "./ingested_files.json"
DOCS_PATH = "./docs"
CHROMA_PATH = "./chroma"

def load_tracking():
    if not os.path.exists(TRACKING_FILE):
        return []
    with open(TRACKING_FILE, "r") as f:
        content = f.read().strip()
        if not content:  # file exists but is empty
            return []
        return json.loads(content)

def save_tracking(ingested_files):
    with open(TRACKING_FILE, "w") as f:
        json.dump(ingested_files, f, indent=2)

def get_new_files():
    already_ingested = load_tracking()
    all_files = [os.path.join(DOCS_PATH, f)
    for f in os.listdir(DOCS_PATH)
    if f.endswith(".pdf")]
    new_files = [f for f in all_files if f not in already_ingested]
    return new_files, already_ingested



#DirectoryLoader reads all documents
#So far only works for pdfs
def load_documents(files):
    documents = []
    for file in files:
        loader = PyPDFLoader(file)
        documents.extend(loader.load())
    print(f"Loaded {len(documents)} documents page(s) from {len(files)} new files")
    return documents
    # loader = DirectoryLoader("./docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
    # documents = loader.load()
    # print(f"Loaded {len(documents)} document(s)")
    # return documents


#RecursiveCharacterTextSplitter with a chuck_size of 1024
#Chunk size of 500 might be more accurate
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

#OllamaEmbeddings with nomic embed text
def create_embeddings():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

#Persist to disk
def store_in_chromadb(chunks,embeddings):
    # vectorstore = Chroma.from_documents(
    #     documents=chunks,
    #     embedding=embeddings,
    #     persist_directory="./chroma_db"
    # )
    # print("Vector store saved to /chroma_db")
    # return vectorstore
    # If DB already exists, add to it — don't wipe it
    if os.path.exists(CHROMA_PATH):
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        vectorstore.add_documents(chunks)
        print("Added new chunks to existing vector store")
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        print("Created new vector store")
    return vectorstore

#Chain all processes together
def main():
    print("starting processes")
    # docs = load_documents()
    # chunks = split_documents(docs)
    # embeddings = create_embeddings()
    # store_in_chromadb(chunks,embeddings)
    new_files, already_ingested = get_new_files()

    if not new_files:
        print("No new documents found. Nothing to do.")
        return

    print(f"Found {len(new_files)} new file(s): {new_files}")

    docs = load_documents(new_files)
    chunks = split_documents(docs)
    embeddings = create_embeddings()
    store_in_chromadb(chunks, embeddings)

    # Save updated tracking list
    updated = already_ingested + new_files
    save_tracking(updated)
    print("Finished processes")

if __name__ == "__main__":
    main()
