import os
import tempfile
import traceback
import uuid
import torch
import requests
import fitz  # PyMuPDF
from PIL import Image
from typing import List
import chromadb
from embedding import get_jina_embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize global variables
chroma_client = chromadb.PersistentClient(path=r"C:\Users\WIN\Desktop\rag")
collection_name = str(uuid.uuid4())


def download_pdf(url: str) -> str:
    """
    Downloads a PDF file from the given URL and saves it to a temporary file.

    Args:
        url (str): The URL of the PDF file to download.

    Returns:
        str: The path of the temporary file where the PDF was saved.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        with open(temp_file.name, 'wb') as f:
            f.write(response.content)
        return temp_file.name
    except Exception as e:
        print(f"Failed to download PDF: {traceback.format_exc(e)}")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file located at the given `pdf_path`.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF file.
    """
    try:
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        document.close()
        return text
    except Exception as e:
        print(f"Failed to extract text from PDF: {traceback.format_exc(e)}")


def create_chunk(text: str, chunk_size: int = 2000, chunk_overlap: int = 100) -> List[str]:
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
        docs = text_splitter.create_documents([text])
        chunks = [doc.page_content for doc in docs]
        print(f"Length of chunks : {len(chunks)}")
        return chunks
    except Exception as e:
        print(f"Failed to create chunks: {traceback.format_exc()}")
        raise


def add_data_to_chroma(documents: List[str], name: str):
    """
    Adds a list of documents to a ChromaDB collection with the given name.

    Args:
        documents (List[str]): A list of documents to be added to the collection.
        name (str): The name of the collection to add the documents to.

    Returns:
        Tuple[chromadb.Collection, str]: A tuple containing the updated collection and its name.

    """
    try:
        db = chroma_client.get_collection(name=name)
        for doc_id, document in enumerate(documents):
            embeddings = get_jina_embeddings([document])
            db.add(documents=[document], ids=[
                   str(doc_id)], embeddings=embeddings)
        return db, name
    except Exception as e:
        print(f"Failed to create ChromaDB: {traceback.format_exc()}")


def get_relevant_data(query: str, n_results: int, collection_name: str) -> List[str]:
    """
    Retrieves relevant data from a collection based on a query and returns the top N results.

    Args:
        query (str): The query string to search for relevant data.
        n_results (int): The number of top results to return.
        collection_name (str): The name of the collection to search in.

    Returns:
        List[str]: A list of the top N relevant documents.

    """
    try:
        db = chroma_client.get_collection(name=collection_name)
        embeddings = get_jina_embeddings(text_list=[query])
        results = db.query(query_embeddings=embeddings, n_results=n_results)
        return [doc[0] for doc in results['documents']]
    except Exception as e:
        print(
            f"Failed to retrieve relevant passages: {traceback.format_exc()}")


def add_to_vectordb(pdf_url: str):
    """
    Adds a PDF document to a ChromaDB collection.

    Args:
        pdf_url (str): The URL of the PDF document to add.

    Returns:
        dict: A dictionary indicating the success of the operation.
    """
    try:
        # get list of collection
        colls = chroma_client.list_collections()
        collection_list = [coll.name for coll in colls]
        if collection_name not in collection_list:
            chroma_client.create_collection(name=collection_name)

        collection = chroma_client.get_collection(collection_name)
        print(collection)

        # Download and process PDF
        pdf_path = download_pdf(url=pdf_url)
        text = extract_text_from_pdf(pdf_path=pdf_path)
        text_list = create_chunk(text=text)

        # Add data to ChromaDB
        add_data_to_chroma(documents=text_list, name=collection_name)
        return {"success": True}

    except Exception as e:
        print(f"An error occured : {traceback.format_exc()}")
