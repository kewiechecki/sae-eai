from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch as t
import os
from sparsify import Sae

with open("flowcharts.txt", "r", encoding="utf-8") as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(text)
text_lines = [chunk for chunk in chunks]

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

#out = "embeddings/" + model_id + "/"
out = "features/" + sae_id + "/"
os.makedirs(out, exist_ok=True)
