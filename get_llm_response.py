
from transformers import AutoModelForCausalLM, AutoConfig
from transformers import AutoTokenizer, Phi3ForCausalLM
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from helpers import get_relevant_data, collection_name


def make_rag_prompt(query: str, relevant_passage: str) -> str:
    try:
        escaped_passage = relevant_passage.replace(
            "'", "").replace('"', "").replace("\n", " ")
        prompt = f"""
            You are an informative bot equipped to answer questions using the text from the provided reference passage. Your expertise includes a thorough understanding of the Transformer model, which is noted for its unique architecture tailored to sequence transduction tasks. This model exclusively employs self-attention mechanisms, eliminating traditional recurrence and convolutions, thereby improving parallelization and performance significantly.

            **QUESTION:** '{query}'

            **PASSAGE:** '{escaped_passage}'

            **ANSWER:**
            """

        return prompt
    except Exception as e:
        print(f"An error occured : {traceback.format_exc()}")


def generate_response(prompt: str, model_name: str = "microsoft/Phi-3-vision-128k-instruct") -> str:
    """
    Generates a response to a given prompt using a text generation model.

    Args:
        prompt (str): The prompt for generating the response.
        model_name (str, optional): The name of the text generation model to use. Defaults to "microsoft/Phi-3-vision-128k-instruct".

    Returns:
        str: The generated response.

    """
    try:

        model = AutoModelForCausalLM.from_pretrained(
            model_name,  trust_remote_code=True)
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipe = pipeline("text-generation", model=model,
                        tokenizer=tokenizer, device="cpu")
        generation_args = {"max_new_tokens": 500,
                           "return_full_text": False, "temperature": 0.0, "do_sample": False}
        output = pipe([{"role": "user", "content": prompt}], **generation_args)
        answer = output[0]['generated_text']
        return answer
    except Exception as e:
        print(f"Failed to generate answer: {traceback.format_exc()}")


def modify_user_query(user_query: str):
    """
    Modifies the given user query by generating a response that rewrites the original query using a clear, technically sound, well-structured, and persuasive first-person narrative.

    Parameters:
        user_query (str): The original user query to be modified.

    Returns:
        str: The modified user query, or the original user query if no response is generated.

    """
    try:
        prompt = f"""You are skilled at refining user queries. Please rewrite the original query using a clear, technically sound, well-structured, and persuasive first-person narrative.

                Original Query: {user_query}"""

        response = generate_response(prompt=prompt)
        if response:
            return response
        else:
            return user_query
    except Exception as e:
        print(f"An error occured : {traceback.format_exc()}")


print(modify_user_query(user_query="what is transformers?"))


def get_response_from_llm(query: str):
    """
    Retrieves relevant data based on a query, generates a prompt using the relevant data, and generates a response using a text generation model.

    Args:
        query (str): The query string to search for relevant data.

    Returns:
        str: The generated response.
    """

    try:
        modified_query = modify_user_query(user_query=query)
        print(f"Modified user query is  : {modified_query}")
        # get relevant data from vectordatabase
        results = get_relevant_data(
            query=modified_query, n_results=5, collection_name=collection_name)
        relevant_data = " ".join(results)
        prompt = make_rag_prompt(query=query, relevant_passage=relevant_data)
        output = generate_response(prompt=prompt)

        print("Generated Answer:", output)
        return {"response": output}

    except Exception as e:
        print(f"An error occurred: {traceback.format_exc()}")


# model = Phi3ForCausalLM.from_pretrained(
#     "microsoft/phi-3-mini-4k-instruct", use_safetensors=True)
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")

# prompt = "This is an example script ."
# inputs = tokenizer(prompt, return_tensors="pt")

# # Generate
# generate_ids = model.generate(inputs.input_ids, max_length=30)
# tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
#                        clean_up_tokenization_spaces=False)[0]
