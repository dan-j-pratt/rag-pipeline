from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain



#Load persisted ChromaDB from disk
def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    print("Vectors loaded from chroma")
    return vectorstore

#vectorstore.as_retriever(k=3)
def build_retriever(vectorstore):
    #when the user asks a question, the retriever searches ChromaDB and pulls the most
    #similar chunks
    retriever = vectorstore.as_retriever(
        #search_type can be swapped for mmr or similarity_score_threshold
        #"similarity" is a cosine similarity search
        search_type="similarity",
        #mmr avoids returning similar 3 chunks that all say the same thing, use when documents are prone to repeating
        #search_type="mmr",
        #sst only returns chunks who score above a certain  score threshold
        #search_type"similarity_score_threshold",

        #the k value is the k most relevant chunks from stored documents
        search_kwargs={"k":3})
        #swap out for score_threshold when using similarity_score_threshold
        #search_kwargs={"score_threshold":0.7,"k":3})
    return retriever

#inject context chunks + user question
def build_prompt_template():
    #For the template, you can swap out for other personas as well:
    #You are a patient tutor. Use the lecture notes below to answer the question. Explain clearly as if teaching a
    #student
    template = """
    You are an assistant. Use the context below to answer the question. If you are unable to answer from the context,
    say "I dont have enough information"
    
    Context:{context}
    
    Question:{question}
    
    Answer
    """
    prompt = PromptTemplate(template=template,input_variables=["context","question"])
    return prompt

#langchain Retriebalqa chain with Ollama LLM
def build_chain(retriever, prompt):
    llm = ChatOllama(model="llama3")
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        #chain_type="map_reduce" #Sends first chunk, then refines answer with following chunks (Large docs)
                                        #chain_type="refine" #Sends first chunk, then iteratively refines the answer with each additional chunk (quality>speed)
                                        #chain_type="map_rerank"  #Sends each chunk to LLM seperately, then combines answers (most confident single answer)
                                        chain_type="stuff", #Stuffs all chunks in one prompt (small docs and few chunks)
                                        retriever=retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":prompt}
                                        )
    return chain

#Run the chain and return answer and sources
def query(question):
    vectorstore = load_vectorstore()
    retriever = build_retriever(vectorstore)
    prompt = build_prompt_template()
    chain = build_chain(retriever, prompt)
    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = [doc.metadata["source"] for doc in result["source_documents"]]
    return answer,sources

if __name__ == "__main__":
    question = "What is this document about?"
    answer, sources = query(question)
    print(f"\nAnswer: {answer}")
    print(f"\nSources: {sources}")