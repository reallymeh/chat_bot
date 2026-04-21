from config import load_config

from langchain_ollama import OllamaEmbeddings, ChatOllama

config = load_config()

def get_embeddings():
    return OllamaEmbeddings(
    model=config["embedding_model"]
)

def get_llm():
    return ChatOllama(
    model=config["llm_model"],      
    temperature=config["temperature"],
    num_predict=config["max_tokens"]
)
