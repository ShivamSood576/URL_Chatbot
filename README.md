
# RAG-Based Question Answering API using Flask and LangChain 📝📚

## Overview

This project is a **Retrieval-Augmented Generation (RAG)** system built with Flask, LangChain, FAISS, and Google's Gemini AI models. The API allows users to query a set of documents retrieved from a **specified URL** and receive AI-generated answers.
## Features 🚀

- **Flask API** for handling user queries
- **UnstructuredURLLoader** to extract data from web pages
- **Chat Interface:** Ask questions about the uploaded PDF content in a user-friendly chat interface.
- **RAG Pipeline:** Utilizes LangChain for document splitting, embeddings creation, and retrieval.
- **Contextual Answers:** Provides accurate answers based on the uploaded document.
- **Conversation History:** Saves Q&A sessions with timestamps in a CSV file.
- **Streamlit Integration:** A seamless and intuitive user experience.
- **Google Generative AI Integration:** For generating embeddings and AI-based responses.

## Tech Stack 🛠️

- **Python**: Core programming language.
- **Streamlit**: For creating the web application interface.
- **LangChain**: For the RAG pipeline and LLM integration.
- **Google Generative AI**: For embeddings and chat responses.
- **FAISS**: For vector similarity search and retrieval.
- **Flask**: for API


  ## How to Run the Project 🏃‍♂️

### Prerequisites

1. **Install Python 3.8+**
2. Install the required libraries by running:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your **Google Generative AI credentials**:

   ```env
   GOOGLE_API_KEY=your_google_api_key
   ```


### Steps to Run

1. Clone the repository:

   ```bash
   [git clone https://github.com/your-repo-name/pdf-chat-app.git]
   cd RAG-based_Chatbot

2. Start the application:

   ```bash
    run app.py
   ```

# API Usage

## Endpoint: `/ask`
- **Method:** `POST`
- **Content-Type:** `application/json`

### Request Body:
```json
{
    "query": "What courses are available on Brainlox?"
}


## Output🙌

![image](https://github.com/user-attachments/assets/50a8b12f-d3e8-4e63-bc52-830ccfd9587b)





