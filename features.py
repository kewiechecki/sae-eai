from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch as t
import os
from sparsify import Sae

def embed(model, sae, inputs):
    with t.no_grad():
        inputs = inputs.to(device)
        outputs = model(**inputs, output_hidden_states=True)

        latent_acts = []
        for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
            sae = sae.to(device)
            # (N, D) input shape expected
            hidden_state = hidden_state.flatten(0, 1)
            latent_acts.append(sae.encode(hidden_state))


with open("flowcharts.txt", "r", encoding="utf-8") as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(text)
text_lines = [chunk for chunk in chunks]

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
sae_id = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_id, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

saes = Sae.load_many(sae_id)

model = model.to(device)
saes = saes.to(device)
inputs = [tokenizer(txt, return_tensors="pt") for txt in chunks]

#out = "embeddings/" + model_id + "/"
out = "features/" + sae_id + "/"
os.makedirs(out, exist_ok=True)
F = [embed(model, saes, x) for x in inputs]
