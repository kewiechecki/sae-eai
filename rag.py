from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
import os
from sparsify import Sae

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
sae_id = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"

model = AutoModelForCausalLM.from_pretrained(model_id, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

saes = Sae.load_many(sae_id)

with open("flowcharts.txt", "r", encoding="utf-8") as f:
    text = f.read()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(text)



from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def emb_text(text):
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])

text_lines = [chunk for chunk in chunks]

from pymilvus import MilvusClient

milvus_client = MilvusClient(uri="./hf_milvus_demo.db")

collection_name = "rag_collection"

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
)

from tqdm import tqdm

data = []

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": emb_text(line), "text": line})

insert_res = milvus_client.insert(collection_name=collection_name, data=data)
insert_res["insert_count"]

question = "What is the legal basis for the proposal?"

search_res = milvus_client.search(
    collection_name=collection_name,
    data=[emb_text(question)],  # Use the `emb_text` function to convert the question to an embedding vector
    limit=3,  # Return top 3 results
    search_params={"metric_type": "IP", "params": {}},  # Inner product distance
    output_fields=["text"],  # Return the text field
)

import json

retrieved_lines_with_distances = [(res["entity"]["text"], res["distance"]) for res in search_res[0]]
print(json.dumps(retrieved_lines_with_distances, indent=4))

context = "\n".join([line_with_distance[0] for line_with_distance in retrieved_lines_with_distances])
PROMPT = """
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""
prompt = PROMPT.format(context=context, question=question)
