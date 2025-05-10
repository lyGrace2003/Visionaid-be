import json
import openai
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
from django.conf import settings

api_key = settings.OPENAI_API_KEY

room_data = {
    "kitchen": ["microwave", "oven", "toaster", "sink", "fork", "spoon", "knife", "bottle", "cup", "refrigerator", "dining table"],
    "bathroom": ["toilet", "sink", "hair drier", "toothbrush"],
    "bedroom": ["bed", "laptop", "book", "teddy bear", "clock"],
    "living room": ["couch", "tv", "potted plant", "clock"],
    "office": ["laptop", "book", "keyboard", "mouse", "monitor", "chair"],
    "dining room": ["dining table", "chair", "cup"],
    "any": ["person", "chair"]
}

client = chromadb.Client(Settings())
collection = client.get_or_create_collection(name="room_db")

#Init Embedding Function 
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-small"
)

#convert list to vector
def get_embedding(text: str):
    return openai_ef(text)[0]  # openai_ef returns a list of numpy arrays; we return the first and only one

#Store to Vector DB
def populate_vector_store_if_empty():
    if collection.count() == 0:
        print("Embedding and storing room data...")
        for i, (room, objects) in enumerate(room_data.items()):
            doc = f"Room: {room}. Objects commonly found: {', '.join(objects)}."
            embedding = get_embedding(doc).tolist()  # convert NumPy array to list
            collection.add(
                documents=[doc],
                embeddings=[embedding],
                ids=[f"room-{i}"]
            )
        print("Vector DB populated.")
    else:
        print("Vector DB already populated. Skipping embedding.")
