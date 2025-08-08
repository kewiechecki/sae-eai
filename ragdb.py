from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from tqdm import tqdm

PROMPT = """
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""

class RAGDB:
    def __init__(
            self,
            db_file,
            model="BAAI/bge-small-en-v1.5",
            chunk_size=1000,
            chunk_overlap=200,
            metric_type="IP",  # Inner product distance
            consistency_level="Strong",  # Strong consistency level
            ):
        self.db_file = Path(db_file)
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metric_type = metric_type
        self.consistency_level = consistency_level

        self.embedding_model = SentenceTransformer(model)

        self.embedding_dim = len(self.encode("test"))
        self.db = MilvusClient(uri=str(self.db_file))

        #if self.db_file.is_file():

    def encode(self, text):
        return self.embedding_model.encode(
                [text], normalize_embeddings=True
                ).tolist()[0]

    def create(self, collection):
        self.db.create_collection(
            collection_name=collection,
            dimension=self.embedding_dim,
            metric_type=self.metric_type,
            consistency_level=self.consistency_level,
        )

    def insert(self, collection, text):
        with open(txt, "r", encoding="utf-8") as f:
            text = f.read()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                       chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text)

        data = []

        for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
            data.append({"id": i, "vector": emb_text(line), "text": line})

        insert_res = milvus_client.insert(collection_name=collection_name, data=data)
        insert_res["insert_count"]

    def rag(self, collection, question):
        search_res = self.db.search(
            collection_name=collection_name,
            data=[emb_text(question)],  # Use the `emb_text` function to convert the question to an embedding vector
            limit=3,  # Return top 3 results
            search_params={"metric_type": self.metric_type, "params": {}},  # Inner product distance
            output_fields=["text"],  # Return the text field
        )

        retrieved_lines_with_distances = [
                (res["entity"]["text"], res["distance"]) for res in search_res[0]
                ]
        context = "\n".join(
            [line_with_distance[0] 
             for line_with_distance in retrieved_lines_with_distances]
            )
        prompt = PROMPT.format(context=context, question=question)
        return prompt
