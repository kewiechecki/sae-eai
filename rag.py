from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def emb_text(text):
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

from pymilvus import MilvusClient

milvus_client = MilvusClient(uri="./hf_milvus_demo.db")

collection_name = "rag_collection"

PROMPT = """
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""

def rag(question):
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[emb_text(question)],  # Use the `emb_text` function to convert the question to an embedding vector
        limit=3,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )

    retrieved_lines_with_distances = [(res["entity"]["text"], res["distance"]) for res in search_res[0]]
    context = "\n".join([line_with_distance[0] for line_with_distance in retrieved_lines_with_distances])
    prompt = PROMPT.format(context=context, question=question)
    return prompt
