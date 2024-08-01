
from transformers import AutoModel


def get_jina_embeddings(text_list: list):
    model = AutoModel.from_pretrained(
        'jinaai/jina-clip-v1', trust_remote_code=True)
    text_embeddings = model.encode_text(text_list)
    return text_embeddings
