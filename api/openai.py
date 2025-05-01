import openai
import json
from django.conf import settings

api_key = settings.OPENAI_API_KEY 

def generate_scene_description(detected_objects):
    prompt = f"""
    Given the following detected items and their bounding boxes, generate a clear, natural paragraph describing the scene,
    designed to assist a visually impaired individual.

    Instructions:
    - Items with label "ocr" represent text detected in the scene (e.g., signs, posters, or labels). These are not physical objects, but text that should be *read* or described as part of a sign or written information.
    - Items with other labels represent physical objects.
    - Focus on describing only the most relevant and clearly visible physical objects. Ignore distant, small, or unimportant ones.
    - Mention the content of OCR text naturally, such as “a sign reads 'Caution'" or “a banner says 'Welcome Students'".
    - Use spatial terms such as 'to the left', 'in front of', 'behind', 'near', and 'far' to describe object positions relative to one another.
    - Do NOT mention any coordinates, bounding boxes, numbers, or percentages unless it is part of the OCR text.
    - Start with the most prominent features of the scene, then describe other relevant objects based on their spatial relationships.
    - Write the description as a concise and natural-sounding paragraph. Avoid listing items mechanically or including detection metadata like confidence scores.

    Each item contains:
    - label: 'ocr' (if it is detected text) or a physical object class
    - box: [x1, y1, x2, y2] image coordinates
    - ocr_text: (if available, contains the detected text and confidence)

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