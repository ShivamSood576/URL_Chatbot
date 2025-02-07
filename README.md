# RAG-Based Question Answering API using Flask and LangChain

## Overview

This project is a Retrieval-Augmented Generation (RAG) system built with Flask, LangChain, FAISS, and Google's Gemini AI models. The API allows users to query a set of documents retrieved from a specified URL and receive AI-generated answers.

## Features

- **Flask API** for handling user queries
- **UnstructuredURLLoader** to extract data from web pages
- **RecursiveCharacterTextSplitter** for efficient text chunking
- **FAISS** vector store for document retrieval
- **GoogleGenerativeAIEmbeddings** for vector representations
- **ChatGoogleGenerativeAI** for AI-based responses
- **Retrieval-Augmented Generation (RAG) pipeline** for improved responses

---

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install Flask python-dotenv langchain langchain_community langchain_google_genai faiss-cpu requests

You also need a Google API key to use the GoogleGenerativeAIEmbeddings and ChatGoogleGenerativeAI models. Store it in a .env file:

GOOGLE_API_KEY=your_api_key_here

Setup
Clone this repository:
git clone https://github.com/your-username/your-repo.git
cd your-repo
Install dependencies:

pip install -r requirements.txt
Set up environment variables:

Create a .env file in the project directory

Add the following line to the .env file:

GOOGLE_API_KEY=your_api_key_here
Run the Flask app:

python app.py
