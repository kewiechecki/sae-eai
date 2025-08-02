from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
import os

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
sae_id = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"
layer_id = 0
device = t.device("cuda")


model = AutoModelForCausalLM.from_pretrained(model_id, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

embed_file = "embeddings/" + model_id +"/" + str(layer_id)+".pt"
hookpt = "layers."+str(layer_id)+".mlp"
sae = Sae.load_from_hub(sae_id, hookpoint=hookpt)

model = model.to(device)
sae = sae.to(device)

#out = "embeddings/" + model_id + "/"
out = "features/" + sae_id + "/"
os.makedirs(out, exist_ok=True)
embed_file = out + str(layer_id)+".pt"

text = "Hello, world!"
inputs = tokenizer(text, return_tensors="pt")
inputs = inputs.to(device)
outputs = model(**inputs)

# outputs.hidden_states is a tuple: (embedding_output, layer1, ..., layerN)
layer_id = 0  # example: get activations after layer 10
layer_embedding = outputs.hidden_states[layer_id]  # shape: (batch, seq_len, hidden_dim)

# Save to file
t.save(layer_embedding, out + str(layer_id) + ".pt")

def embed_layer(model, sae, layer, inputs):
    #with t.inference_mode():
    with t.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[0].flatten(0, 1)
        return sae.encode(hidden_state)

def unembed_layer(model, sae, layer, top_acts, top_inds):
    with t.no_grad():
        n_layers = len(model.model.layers)
        E = sae.decode(top_acts, top_inds)
        E = E.unsqueeze(0)
        seq_len =E.shape[1]


        # Typically, position_ids is [batch, seq_len]
        position_ids = t.arange(seq_len, dtype=t.long, device=device).unsqueeze(0)

        # attention_mask: usually all ones, unless you want masking
        attention_mask = t.ones((1, seq_len), dtype=t.bool, device=device)

        layers = range(layer, n_layers)
        for i in layers:
            # Get the relevant block and call with all required arguments:
            block = model.model.layers[i]
            out = block(
                E,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden = out[0] if isinstance(out, (tuple, list)) else out
        return hidden

F = embed_layer(model, tokenizer, sae, text, layer_id)
with t.no_grad():
    E = sae.decode(F.top_acts, F.top_indices)
    model.model.layers[0](E)

E = unembed_layer(model, sae, layer_id+1, F.top_acts, F.top_indices)

