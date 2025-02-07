from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

app = Flask(__name__)

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

# Create Chains
question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route('/ask', methods=['POST'])
def ask():
    """
    API endpoint to handle user queries and return RAG-based responses.
    Expected input: JSON {"query": "your question here"}
    """
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    response = rag_chain.invoke({"input": query})
    answer = response.get("answer", "I couldn't generate an answer.")

    return jsonify({"question": query, "answer": answer})

if __name__ == '__main__':
    app.run(debug=True)

import requests

# url = "http://127.0.0.1:5000/ask"
# data = {"query": "What courses are available on Brainlox?"}

# response = requests.post(url, json=data)
# print(response.json())
