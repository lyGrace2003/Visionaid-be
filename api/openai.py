import openai
import json
from django.conf import settings

api_key = settings.OPENAI_API_KEY 

def generate_scene_description(detected_objects):
    prompt = f"""
    Given the following detected objects and their bounding boxes, generate a clear, natural paragraph describing the scene,
    designed to assist a visually impaired individual.

    Instructions:
    - Focus on describing only the most relevant and clearly visible objects. Ignore distant, small, or unimportant objects.
    - Use spatial terms such as 'to the left', 'in front of', 'behind', 'near', and 'far' to describe object positions relative to one another.
    - Do NOT mention any coordinates, bounding boxes, numbers, or percentages unless it is text detected from OCR.
    - If an object contains detected text, mention the text naturally in the description.
    - Start with the most prominent features of the scene, then describe other relevant objects based on their spatial relationships.
    - Write the description as a concise and natural-sounding paragraph without mentioning confidence scores, coordinates, or detection statistics.

    Each object contains:
    - label (object class)
    - box ([x1, y1, x2, y2] image coordinates)
    - ocr_text (if available)

    Detected Objects:
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