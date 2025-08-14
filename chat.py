from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from ragdb import RAGDB

'''
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
'''

sys_prompt = "You are an automated diagnostician. When presented with symptoms, pull up the appropriate diagnostic workflows and then ask the user follow up questions about their symptoms until you have enough information to make a recommendation or diagnosis."
hyperparams = {
        "max_new_tokens" : 1024,
        "do_sample" : True,
        "temperature" : 0.7,
        "top_p" : 0.95
        }

def embed_layer(model, sae, layer, inputs):
    #with t.inference_mode():
    with t.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[0].flatten(0, 1)
        return sae.encode(hidden_state)

class Chat:
    def __init__(
            self,
            featurized,
            ragdb,
            collection,
            sys_prompt = sys_prompt,
            hyperparams = hyperparams
            ):
        self.featurized = featurized
        self.model = featurized.model
        self.saes = featurized.saes

        self.db = ragdb
        self.collection = collection

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.saes = Sae.load_many(sae_id)
        self.n_layers = len(self.saes)

        self.history = [sys_prompt]

    def tokenize(self, prompt):
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def update_chat(self, user_input):
        user_input = self.db.rag(self.collection, user_input)
        self.history.append(f"User: {user_input}")
        prompt = "\n".join(history) + "\nAssistant:"

        inputs = self.tokenize(prompt)
        output = self.model.generate(
            **inputs,
            **hyperparams,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id  # for models without a pad token
        )
        response = self.tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print("Assistant:", response.strip())
        self.history.append(f"Assistant: {response.strip()}")

    def eval(self, layer):
        sae = self.saes[layer].to(self.device)
        prompt = self.history[len(self.history)]
        inputs = self.tokenize(prompt)
        return embed_layer(self.model, sae, layer, inputs)

    def run(self):
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if user_input.lower() in ("eval", "audit"):
                print("Select layer to audit (0-" + str(self.n_layers) + "):")
                user_input = input("User: ").strip()
                if int(user_input) in range(n_layers):
                    break
                else:
                    print("Invalid layer.")
                    continue
            self.update_chat(user_input)

