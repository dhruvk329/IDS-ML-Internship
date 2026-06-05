#All API keys and related information is removed since code is public
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

os.environ["XAI_API_KEY"] = ""

def load_documents(folder_path):
    loader = DirectoryLoader(
        folder_path,
        glob="",
        loader_cls=TextLoader
    )
    return loader.load()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)

def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    store = Chroma.from_documents(chunks, embeddings, persist_directory="db")
    return store

def build_qa_chain(store):
    retriever = store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(
        model="grok-2-latest",
        base_url="",
        api_key=os.environ[""],
        temperature=0
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain

def main():
    documents = load_documents("client_archives")
    chunks = split_documents(documents)
    store = build_vector_store(chunks)
    qa = build_qa_chain(store)

    while True:
        question = input("Ask a question (or type quit): ")
        if question.lower() == "quit":
            break

        result = qa.invoke({"query": question})
        print("\nAnswer:")
        print(result["result"])

        print("\nSources:")
        for doc in result["source_documents"]:
            print(doc.metadata.get("source", "unknown"))
        print()

if __name__ == "__main__":
    main()
