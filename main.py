
import platform
import traceback
from fastapi import FastAPI, Request

from get_llm_response import get_response_from_llm
from helpers import add_to_vectordb

app = FastAPI()

# pdf_url = "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"


@app.post("/add_to_vdb")
async def add_to_collection(request: Request):
    try:
        data = await request.json()
        print(data)
        url = data.get("url")
        response = add_to_vectordb(pdf_url=url)
        if response:
            print(f"Successfully added to vector db")
            return response
        else:
            print(f"Failed to add vector db")
            return None

    except Exception as e:
        print(f"An error occurred: {traceback.format_exc()}")


@app.post("/get_response")
async def get_response_phi3(request: Request):
    try:
        data = await request.json()
        print(data)
        query = data.get("query")
        response = get_response_from_llm(query=query)
        print(f"Response = {response}")
        if response:
            print(f"Successfully generated response")
            return response
        else:
            print(f"Failed to generate response")
            return None
    except Exception as e:
        print(f"An error occurred: {traceback.format_exc()}")


if __name__ == "__main__":
    if platform.system() == "Windows":
        import uvicorn

        uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
