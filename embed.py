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

inputs = [tokenizer(txt, return_tensors="pt") for txt in chunks]

from pymilvus import MilvusClient

milvus_client = MilvusClient(uri="./hf_milvus_demo.db")

collection_name = "rag_collection"

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
)

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

text_lines = [chunk.page_content for chunk in chunks]

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def emb_text(text):
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

#out = "embeddings/" + model_id + "/"
out = "features/" + sae_id + "/"
os.makedirs(out, exist_ok=True)

text = "Hello, world!"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = inputs.to(device)
outputs = model(**inputs, output_hidden_states=True)

# outputs.hidden_states is a tuple: (embedding_output, layer1, ..., layerN)
layer_id = 0  # example: get activations after layer 10
layer_embedding = outputs.hidden_states[layer_id]  # shape: (batch, seq_len, hidden_dim)

# Save to file
t.save(layer_embedding, out + str(layer_id) + ".pt")

def embed(model, tokenizer, sae, inputs):
    with t.inference_mode():
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)

        latent_acts = []
        for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
            # (N, D) input shape expected
            hidden_state = hidden_state.flatten(0, 1)
            latent_acts.append(sae.encode(hidden_state))

with t.inference_mode():
    model = AutoModelForCausalLM.from_pretrained(model_id)
    outputs = model(**inputs, output_hidden_states=True)

    latent_acts = []
    for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
        # (N, D) input shape expected
        hidden_state = hidden_state.flatten(0, 1)
        latent_acts.append(sae.encode(hidden_state))


