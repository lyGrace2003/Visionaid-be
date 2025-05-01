import openai
import json
from django.conf import settings

api_key = settings.OPENAI_API_KEY 

def generate_scene_description(detected_objects):
    prompt = f"""
    You are given a list of detected items in a scene captured by a camera. Your task is to generate a clear, helpful, and natural paragraph describing the scene for a visually impaired user.

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
    {json.dumps(detected_objects)}
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