from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
import os
from sparsify import Sae

import math

def rope(head_dim, seq_len, n_heads, offset=0, dtype=t.float32, device=None):
    """
    Return cos/sin position embeddings of shape (seq_len, head_dim).
    Used for RoPE (Rotary Positional Embedding).
    """
    theta = 10000.0 ** (-t.arange(0, head_dim, 2, dtype=dtype, device=device) / head_dim)
    seq = t.arange(offset, offset + seq_len, dtype=dtype, device=device)
    #freqs = t.einsum("i,j->ij", seq, theta)  # (seq_len, head_dim/2)
    freqs = t.outer(seq, theta)  # (seq_len, head_dim // 2)

    # concat even and odd sine/cosine components
    emb = t.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    emb = emb[None, :, None, :].expand(1, seq_len, n_heads, head_dim)  # broadcast across heads
    return emb.cos(), emb.sin()

def layer_args(model, inputs):
    x = inputs['input_ids']
    attention_mask = inputs['attention_mask'].to(t.bool)
    seq_len = x.shape[1]
    # Typically, position_ids is [batch, seq_len]
    position_ids = t.arange(seq_len, dtype=t.long, device=model.device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(x, position_ids)

    # attention_mask: usually all ones, unless you want masking
    #attention_mask = t.ones((1, seq_len), dtype=t.bool, device=model.device)
    #n_heads = model.config.num_attention_heads
    #head_dim = model.config.hidden_size // n_heads
    '''
    position_embeddings = rope(
        head_dim=head_dim,
        seq_len=seq_len,
        n_heads=n_heads,
        offset=0,
        dtype=model.dtype,
        device=model.device
    )
    '''

    return attention_mask, position_embeddings, position_ids

class Featurized:
    def __init__(
            self, model, sae,
            pattern="layers.%s.mlp"):
        self.model_id = model
        self.sae_id = sae
        
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
                model, output_hidden_states=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.n_layers = len(self.model.model.layers)

        self.saes = Sae.load_many(sae, device=t.device("cpu"))
        self.n_saes = len(self.saes)
        self.pattern = pattern
        #self.keys = self.saes.keys()

    def get_sae(self, i):
        if isinstance(i, int):
            i = self.pattern % i
            #i = self.keys[i]
        return self.saes[i].to(self.device)

    def tokenize(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        return inputs.to(self.device)

    def embed_layer(self, layer, inputs):
        #with t.inference_mode():
        sae = self.get_sae(layer)
        with t.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_state = outputs.hidden_states[layer].flatten(0, 1)
            return sae.encode(hidden_state)

    def unembed_layer(self, layer, inputs, features):
        sae = self.get_sae(layer)
        model = self.model
        with t.no_grad():
            E = sae.decode(features.top_acts, features.top_indices)
            E = E.unsqueeze(0)
            seq_len = E.shape[1]

            attention_mask, position_embeddings, position_ids = layer_args(model, inputs)
            layers = range(layer, self.n_layers)
            for i in layers:
                # Get the relevant block and call with all required arguments:
                block = model.model.layers[i]
                out = block(
                    E,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
                E = out[0] if isinstance(out, (tuple, list)) else out
            return E 

    def embed(model, tokenizer, sae, inputs):
        with t.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)

            latent_acts = []
            for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
                # (N, D) input shape expected
                hidden_state = hidden_state.flatten(0, 1)
                latent_acts.append(sae.encode(hidden_state))

