from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rag import rag

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "You are a helpful assistant.\nUser: How do I treat acute abdominal pain?\nAssistant:"

prompt = rag(prompt)

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
output = model.generate(
    **inputs,
    max_new_tokens=256,      # how many tokens to generate in response
    do_sample=True,          # use sampling (for more diversity) or set to False for greedy
    temperature=0.7,         # controls randomness; lower is more deterministic
    top_p=0.95,              # nucleus sampling
    eos_token_id=tokenizer.eos_token_id
)

# Decode
completion = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(completion)
