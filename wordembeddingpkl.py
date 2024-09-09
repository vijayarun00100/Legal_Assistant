import streamlit as st
import os
import pickle
import numpy as np
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

PDF_PATH = "Corpus.pdf"
PERSIST_DIRECTORY = 'corpus_vectorstore'
PICKLE_PATH = 'D://law//embeddings.pkl'
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama2"

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

    def clear(self):
        self.text = ""

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.chat_history = []

def load_embeddings_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        embeddings = pickle.load(f)
        # Ensure embeddings is a NumPy array and convert to list if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        return embeddings.tolist()

class CustomEmbeddingFunction:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.query_to_index = {
            # Example mappings; replace these with actual mappings
            "example_query": 0
        }

    def embed_query(self, query):
        query_index = self.query_to_index.get(query, None)
        if query_index is not None:
            return self.embeddings[query_index]
        else:
            return [0] * len(self.embeddings[0])  # Return a zero vector if not found

    def embed_documents(self, documents):
        # Implement document embedding logic if needed
        pass

def initialize_vectorstore(embedding_function):
    if os.path.exists(PERSIST_DIRECTORY):
        st.text("Loading existing vectorstore...")
        return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_function)
    else:
        st.text("Creating new vectorstore with embeddings...")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function, persist_directory=PERSIST_DIRECTORY)
        vectorstore.persist()
        return vectorstore

def initialize_components():
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template="""You are a knowledgeable chatbot, here to help with questions about the provided document. Your tone should be professional and informative. Provide concise answers.

        Context: {context}
        History: {history}

        User: {question}
        Chatbot:"""
    )
    
    memory = ConversationBufferMemory(memory_key="history", return_messages=True, input_key="question")
    
    return prompt, memory

def main():
    st.title("Law Bot!")

    if not st.session_state.initialized:
        # Load embeddings from pickle file
        embeddings = load_embeddings_from_pickle(PICKLE_PATH)
        embedding_function = CustomEmbeddingFunction(embeddings)
        
        vectorstore = initialize_vectorstore(embedding_function)
        prompt, memory = initialize_components()
        st.session_state.vectorstore = vectorstore
        st.session_state.prompt = prompt
        st.session_state.memory = memory
        st.session_state.initialized = True
        st.text("Chatbot initialized and ready!")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("You:"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            response_container = st.empty()
            stream_handler = StreamHandler(response_container)
            
            llm = Ollama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                callbacks=[stream_handler]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=st.session_state.vectorstore.as_retriever(),
                chain_type_kwargs={
                    "verbose": True,
                    "prompt": st.session_state.prompt,
                    "memory": st.session_state.memory,
                }
            )
            
            response = qa_chain(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response['result']})
            stream_handler.clear()

if __name__ == "__main__":
    main()
