import optparse
import torch as t

from chat import Chat
from featurized import Featurized
from ragdb import RAGDB

temp = 0.7
max_new_tokens = 1024
top_p = 0.95

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
sae_id = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"

embedding_id = "BAAI/bge-small-en-v1.5"
db = "./hf_milvus.db"
text = "flowcharts.md"
collection = "flowcharts"

featurized = Featurized(model_id, sae_id)
inputs = featurized.tokenize("The patient is a 32 year old sexually active female with height 1.7m and weight 70kg. Their symptoms include headache, nausea, and missed period.")
inputs = featurized.tokenize("The patient is a 32 year old marijuana user with height 1.7m and weight 70kg. Their symptoms include headache, nausea, and missed period.")
inputs = featurized.tokenize("The patient is a 32 year old unemployed marijuana user with height 1.7m and weight 70kg. Their symptoms include headache, nausea, and missed period.")
features = featurized.embed_layer(0, inputs)
featurized.unembed_layer(1, inputs, features)

outputs = featurized.model(**inputs, output_hidden_states=True)
E=outputs.hidden_states[0]
E = model.model.norm(E)
logits = model.lm_head(E)

layers = range(1,featurized.n_layers)
with t.no_grad():
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

def main(opts):
    db = RAGDB(opts.db)
    hyperparams = {
            "max_new_tokens" : opts.max_new_tokens,
            "do_sample" : True,
            "temperature" : opts.temp,
            "top_p" : opts.top_p
            }

    if len(opts.insert) > 0: 
        db.insert(opts.collection, opts.insert)
    else: 
        featurized = Featurized(opts.model_id, opts.sae_id)
        chat = Chat(featurized, ragdb)
        chat.run()

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option(
            '-m', '--model', type="str", default=model_id, dest="model_id", 
            help="model name used by AutoModelForCausalLM.from_pretrained"
            )
    parser.add_option(
            '-s', '--sae', type="str", default=model_id, dest="sae_id", 
            help="SAE name used by AutoModelForCausalLM.from_pretrained"
            )
    parser.add_option(
            '-p', '--top-p', type="float", default=temp, dest="top_p", help=""
            )
    parser.add_option(
            '-t', '--temperature', type="float", default=temp, dest="temp", 
            help="temperature"
            )
    parser.add_option(
            '-n', '--max-new-tokens', type="int", default=max_new_tokens,
            dest="max_new_tokens", help="max response length"
            )
    parser.add_option(
            '-d', '--database', type="str", dest='db', default=db, 
            help="database location"
            )
    parser.add_option(
            '-c', '--collection', type="str", dest='collection', default=collection,
            help="database location"
            )
    parser.add_option(
            '-i', '--insert', type="str", dest='text', default="", 
            help="text to insert into database"
            )
    opts, args = parser.parse_args()

    main(opts)
