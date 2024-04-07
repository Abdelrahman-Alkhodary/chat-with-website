import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

load_dotenv()
groq_api = os.getenv("Groq_API_KEY")

def get_vectorstore_from_url(url):
    # Load the document from the website
    loader = WebBaseLoader(url)
    document = loader.load()
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    # Create a vector store from the document chunks
    vector_store = Chroma.from_documents(document_chunks, FastEmbedEmbeddings(cache_dir='./embedding_model/'))
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', '{input}'),
        ('user', "Given the above conversation, generate a search query to look up in order to get information to respond to the user."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    prompt = ChatPromptTemplate.from_messages([
        ("system", 'Answer the user questions based in the below context:\n\n{context}'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', '{input}'),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt=prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_query):
    # create conversation chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversational_rag_chain.invoke({
            'chat_history': st.session_state.chat_history,
            'input': user_query
        })
    return response['answer']


# Set page config
st.set_page_config(page_title="Chat with Websites", page_icon=":shark:", layout="wide")
st.title("Chat with Websites")

# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Enter a website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL in the sidebar.")
else:
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello! I'm a chatbot. How can I help you?")]
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
    
    
    
    # Main content
    # user input
    user_query = st.chat_input("Type a message...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        # add user query to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        # add response to chat history
        st.session_state.chat_history.append(AIMessage(content=response))

    # Conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message('AI'):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('Human'):
                st.write(message.content)