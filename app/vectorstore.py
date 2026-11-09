""" Embedding, ChromaDB, retriever """
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from app.ingestion import docs_list_tokens_split
import os


""" Embedding the documents, storing them in vector db, and creating a retriever to query the vector db """

embedding = FastEmbedEmbeddings()

# Create the db only if it doesn't exist
if not os.path.exists("./chroma_db"):
    vectorstore = Chroma.from_documents(
        documents=docs_list_tokens_split,
        embedding=embedding,
        persist_directory="./chroma_db"
    )
else:
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding
    )

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "lambda_mult": 0.7
    }
)