from config import load_config
from langchain_chroma import Chroma
from helper import get_embeddings, get_llm

from langchain_core.prompts import ChatPromptTemplate

import gradio as gr

config = load_config()
embeddings = get_embeddings()
llm = get_llm()

vector_store = Chroma(
    collection_name=config["chroma_collection"],
    embedding_function= embeddings,
    persist_directory=config["persist_directory"]
)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": config["top_k"]}
)


def stream_response(message, history):
    print(f"Input: {message}. History: {history}\n")

    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge'
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content+"\n\n"


    # make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""

        template = config["system_prompt_template"]
        prompt = ChatPromptTemplate.from_template(template)

        rag_prompt = f"""
        You are an assistant which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge, 
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the provided knowledge.

        The question: {message}

        Conversation history: {history}

        The knowledge: {knowledge}

        """

        print(rag_prompt)

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
chatbot.launch()