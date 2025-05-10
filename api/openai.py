from pathlib import Path
import openai
import json
from collections import defaultdict
from django.conf import settings

from api.vector_store import get_embedding, collection

api_key = settings.OPENAI_API_KEY 

# # Load room knowledge base (RAG retrieval source)
# def load_room_object_map():
#     json_path = settings.BASE_DIR / 'room_objects.json'
#     with open(json_path, "r") as f:
#         return json.load(f)

# room_map = load_room_object_map()
# room_knowledge = json.dumps(room_map, indent=2)

def generate_scene_description(detected_objects):
    detected_objects_str = [obj['label'] for obj in detected_objects]  # Extract only the label
    query_text = f"Detected objects: {', '.join(detected_objects_str)}"

    query_embedding = get_embedding(query_text)

    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    top_docs = "\n".join(results["documents"][0])

    detected_object = json.dumps(detected_objects, indent=2)
    prompt = f"""
    You are given a list of detected items in a scene captured by a camera. Your task is to generate a clear, helpful, and natural paragraph describing the scene for a visually impaired user.

    Here is background knowledge of what objects are typically found in each room:
    
    {top_docs}

    You may use this knowledge **only to infer which room the scene is most likely in**, based on the objects actually detected.

    ### CRITICAL RULES:

    - DO NOT describe any object unless it is explicitly listed in the Detected Items section below.
    - DO NOT add, imagine, or guess objects even if they are commonly found in the inferred room.
    - DO NOT describe the room's furniture, or decorations unless such elements are detected.
    - DO NOT include information based on background knowledge unless it is reflected in the Detected Items.
    - DO NOT simply list objects. Form one fluid, descriptive paragraph.
    - DO NOT include coordinates, scores, or mention objects that are not in the list.
    - DO NOT invent scene context. Stay strictly grounded in the Detected Items.

    ### INSTRUCTIONS:

    - Begin with: “The scene appears to be in [room] based on the detected objects,” if the room is identifiable. Otherwise, say: “The scene's location is unclear based on the detected objects.”
    - Describe only the objects that appear in the Detected Items list.
    - Each detected item includes:
        - `label`: the object's class (e.g., "person", "sign", "bag", or "ocr")
        - `box`: [x1, y1, x2, y2] bounding box (do not include this in output)
        - `ocr_text`: (optional) text in that area
    - If other objects contain `ocr_text`, say: “a [label] reads '[text]'.”
    - If a detected label is "ocr", this is a standalone text. Do **not** call it a sign or banner. Just say, for example: “a text reading 'Welcome' is visible.”
    - Use spatial language when describing object positions (e.g., "to the left", "in front of").
    - Write in one paragraph, natural and informative, but strictly factual.

    ### Detected Items:
    {detected_object}
    """
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates visual scene descriptions."},
                {"role": "user", "content": prompt}
            ]
        )
        scene_description = response.choices[0].message.content
        return scene_description

    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Unable to generate scene description."