sae-eai:
	python -m venv sae-eai
	. sae-eai/bin/activate && pip install git+https://github.com/EleutherAI/sparsify && pip install hf_xet

goodfire:
	python -m venv goodfire
	. goodfire/bin/activate && pip install goodfire

clean:
	rm -rf goodfire
	rm -rf sae-llama
