from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


#DirectoryLoader reads all documents
#So far only works for pdfs
def load_documents():
    loader = DirectoryLoader("./docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s)")
    return documents


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
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Vector store saved to /chroma_db")
    return vectorstore

#Chain all processes together
def main():
    print("starting processes")
    docs = load_documents()
    chunks = split_documents(docs)
    embeddings = create_embeddings()
    store_in_chromadb(chunks,embeddings)
    print("Finished processes")

if __name__ == "__main__":
    main()
