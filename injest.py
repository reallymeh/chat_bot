from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from config import load_config
from helper import get_embeddings

config = load_config()
embedding=get_embeddings()

loader = PyPDFDirectoryLoader(config["data_folder"])  # or support multiple loaders
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["chunk_size"],
    chunk_overlap=config["chunk_overlap"]
)
split_docs = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    collection_name=config["chroma_collection"],
    persist_directory=config["persist_directory"]
)
print("Ingestion complete.")
