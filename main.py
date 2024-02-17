from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_pinecone import Pinecone
from langchain.chains import RetrievalQA

if __name__ == "__main__":
    print("hello world")

    loader = TextLoader("/Users/byronmackay/Dev/AI/udemy-lang-chain-course/intro-to-vector-db/medium-blogs/blog1.txt")
    document = loader.load()
    spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    texts = spliter.split_documents(document)

    embeddings = OpenAIEmbeddings()

    docsearch = Pinecone.from_documents(texts, embeddings, index_name="index237")

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True
    )

    query = "what is a vector database? Give me a 15 word answer for a beginner"
    result = qa({"query": query})
    print(result)
