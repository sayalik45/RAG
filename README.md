## Project Title

RAG-based Question Answering System

### Description

This project is a FASTAPI for a Question Answering system using Retrieval-Augmented Generation (RAG). The system extracts text from a PDF document, processes it into chunks, and stores it in a vector database for efficient retrieval. It then uses a language model to generate responses to user queries based on the relevant passages retrieved from the vector database.

### Features

- **Add to Vector Database**: Extract text from a PDF and add it to the vector database.
- **Get Response**: Retrieve relevant passages from the vector database and generate a response using a language model.

### Directory Structure

```
├── main.py
├── embedding.py
├── get_llm_response.py
├── helpers.py
└── requirements.txt
```

### Prerequisites

- Python 3.11+
- Install required Python packages using pip

### Usage

#### Running the API

Start the FastAPI server: The API will be available at `http://127.0.0.1:8000`.

### API Endpoints

#### 1. Add to Vector Database

- **URL**: `/add_to_vdb`
- **Method**: `POST`
- **Request Body**: JSON object containing the URL of the PDF file to be added.
  ```json
  {
    "url": "https://example.com/sample.pdf"
  }
  ```
- **Response**: JSON object indicating success or failure.

#### 2. Get Response

- **URL**: `/get_response`
- **Method**: `POST`
- **Request Body**: JSON object containing the user query.
  ```json
  {
    "query": "What is the transformers?"
  }
  ```
- **Response**: JSON object containing the generated response.
