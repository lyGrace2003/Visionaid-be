from pathlib import Path
import openai
import json
from collections import defaultdict
from django.conf import settings

api_key = settings.OPENAI_API_KEY 

# Load room knowledge base (RAG retrieval source)
def load_room_object_map():
    json_path = settings.BASE_DIR / 'room_objects.json'
    with open(json_path, "r") as f:
        return json.load(f)

# Converts from room -> objects to object -> rooms
def invert_room_map(room_map):
    obj_to_room = {}
    for room, objects in room_map.items():
        for obj in objects:
            obj_to_room.setdefault(obj, []).append(room)
    return obj_to_room

room_map = load_room_object_map()
OBJECT_TO_ROOM = invert_room_map(room_map)

def guess_room(detected_objects):
    room_counts = {}  # Track the number of objects detected for each room
    
    room_context = {}
    for obj in detected_objects:
        obj_label = obj['label']
        
        if obj_label in OBJECT_TO_ROOM:  # Only consider objects in the mapping
            for room in OBJECT_TO_ROOM[obj_label]:
                room_counts[room] = room_counts.get(room, 0) + 1

    # Handle case where no room is identified
    if not room_counts:
        return "Unknown room"
    
    # Find room with the highest number of objects detected
    most_detected_room = max(room_counts, key=room_counts.get)
    return most_detected_room

def generate_scene_description(detected_objects):
    room = guess_room(detected_objects)
    print("Room:", room)

    prompt = f"""
     You are given a list of detected items in a scene captured by a camera. Your task is to generate a clear, helpful, and natural paragraph describing the scene for a visually impaired user.

    {f"If the room is not 'an unknown room', first state 'The scene appears to be in {room} based on the detected objects' else dont say anything about what kind of room it is"}

    Rules and Instructions:
    - Each item includes:
        - label: the object's class (e.g., "person", "sign", "bag", or "ocr")
        - box: [x1, y1, x2, y2] coordinates of the bounding boxW
        - ocr_text: (optional) text found within that object area
    - Items with label "ocr" represent text detected in the scene that is *not* associated with any physical object. Do **not** call these signs or banners. Instead, mention them only as standalone text, such as “a text reading 'Students' is visible.”
    - If an item has a label other than "ocr" and contains `ocr_text`, assume it is a physical object with text on it (e.g., a sign or label). In this case, describe it naturally like: “a sign reads 'Caution'” or “a banner says 'Welcome'.”
    - **Even if a physical object has no associated text, still describe it if it is clearly visible and relevant** (e.g., “a person is standing near the wall” or “a chair is positioned in front of the desk”).
    - Prioritize clear, spatial descriptions: use terms like 'to the left', 'in front of', 'nearby', 'at the back', etc.
    - Focus only on relevant and clearly visible objects. Ignore clutter or distant items.
    - Do **not** include raw coordinates, bounding box data, or confidence scores in the description.
    - Do not guess objects or infer extra context beyond what is provided.
    - Avoid listing items. Instead, write one natural paragraph that gives an intuitive sense of the scene.

    Detected Items:
    {json.dumps(detected_objects, indent=2)}
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