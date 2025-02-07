import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Streamlit UI Title
st.title("RAG Application")

# Load Data from URL
urls = ['https://brainlox.com/courses/category/technical']
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)

# Vector Store (FAISS) with Google Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(docs, embedding_model)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Language Model (LLM)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# Define System Prompt
system_prompt = (
    "You are an AI assistant specializing in question-answering tasks. "
    "Use the provided context to answer the user's query. "
    "If the answer is not in the context, say: 'I don't know'. "
    "Limit your response to a maximum of three sentences.\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Streamlit Chat Input
query = st.chat_input("Ask something about the website:")

if query:
    # Create Chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Get Response
    response = rag_chain.invoke({"input": query})

    # Display Response
    st.write(response.get("answer", "I couldn't generate an answer."))
