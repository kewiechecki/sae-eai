from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rag import rag

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)


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



history = [
    "You are a helpful assistant."
]

while True:
    user_input = input("User: ").strip()
    if user_input.lower() in ("exit", "quit"):
        break
    user_input = rag(user_input)
    history.append(f"User: {user_input}")
    prompt = "\n".join(history) + "\nAssistant:"

    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id  # for models without a pad token
    )
    response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print("Assistant:", response.strip())
    history.append(f"Assistant: {response.strip()}")
