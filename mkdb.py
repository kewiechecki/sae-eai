with open("flowcharts.txt", "r", encoding="utf-8") as f:
    text = f.read()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(text)



from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def emb_text(text):
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])

text_lines = [chunk for chunk in chunks]

from pymilvus import MilvusClient

milvus_client = MilvusClient(uri="./hf_milvus_demo.db")

collection_name = "rag_collection"

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
)

from tqdm import tqdm

data = []

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": emb_text(line), "text": line})

insert_res = milvus_client.insert(collection_name=collection_name, data=data)
insert_res["insert_count"]


