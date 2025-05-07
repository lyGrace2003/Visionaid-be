import os
import cv2
import numpy as np
import easyocr
import tempfile
import shutil
import onnxruntime as ort
import torch
from ultralytics import YOLO
from pathlib import Path    
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from api.openai import generate_scene_description

from django.conf import settings

model_path = r"C:\Users\Joan\yolov8s.pt"
yolo_model = YOLO(model_path)
reader = easyocr.Reader(['en'])

class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

@csrf_exempt
def upload_image(request):
    """Receives image, performs object detection, applies bounding boxes, and performs OCR."""

    if request.method != 'POST':
        return JsonResponse({'status': 'failed', 'message': 'Invalid request method'}, status=405)

    image = request.FILES.get('image')
    if not image:
        return JsonResponse({'status': 'failed', 'message': 'No image file received'}, status=400)

    temp_dir = Path(tempfile.mkdtemp())
    temp_image_path = temp_dir / image.name

    try:
        # Save uploaded image temporarily
        with open(temp_image_path, 'wb') as temp_file:
            for chunk in image.chunks():
                temp_file.write(chunk)

        # Load image using OpenCV
        original_img = cv2.imread(str(temp_image_path))
        if original_img is None:
            raise Exception("Failed to read image")

        # Perform object detection
        detections = object_detection_view(str(temp_image_path))

        # Annotate image and perform OCR per detection
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            label = detection["label"]
            conf = detection["confidence"]

            # Draw bounding box
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_img, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Perform OCR in ROI
            roi = original_img[y1:y2, x1:x2]
            ocr_result = reader.readtext(roi)
            detection["ocr_text"] = [
                {"text": text, "confidence": float(ocr_conf)}
                for _, text, ocr_conf in ocr_result if ocr_conf > 0.5
            ]

        # Optional: Remove entries with missing label or bad OCR
        detections = [
            det for det in detections
            if det["label"] != Ellipsis and ("ocr_text" not in det or det["ocr_text"] != Ellipsis)
        ]

        # Scene description
        scene_description = generate_scene_description(detections)

        save_scene_description_to_db(scene_description, detections)

        # Save processed image
        upload_dir = Path(settings.MEDIA_ROOT) / 'uploads'
        upload_dir.mkdir(parents=True, exist_ok=True)
        processed_image_path = upload_dir / image.name
        cv2.imwrite(str(processed_image_path), original_img)

        print(f"data: {detections}")
        print(f"image_url: {settings.MEDIA_URL}uploads/{image.name}")

        return JsonResponse({
            "status": "success",
            "message": "Scene Description complete.",
            "scene_description": scene_description,
        })

    except Exception as e:
        return JsonResponse({'status': 'failed', 'message': 'Processing failed', 'error': str(e)}, status=500)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    


def preprocess_image(image_path, img_size=640):
    """Preprocesses the image for YOLOv8."""
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Failed to read image from path")

    # Resize and convert BGR to RGB
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to 0-1 and convert to float32
    img = img.astype(np.float32) / 255.0

    # Transpose to [C, H, W] and add batch dim → [1, C, H, W]
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img

def object_detection_view(image_path):
    """Performs object detection using YOLOv8 model and returns labeled detections."""
    results = yolo_model(image_path)

    raw_detections = []
    for det in results[0].boxes.data.tolist():  # x1, y1, x2, y2, conf, class_id
        x1, y1, x2, y2, conf, class_id = det

        if conf < 0.4:  # Optional: confidence threshold
            continue

        class_id = int(class_id)
        label = class_names[class_id] if class_id < len(class_names) else str(class_id)

        raw_detections.append({
            "label": label,
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": round(conf, 2)
        })

    return raw_detections

from django.db import connection

import json

def save_scene_description_to_db(scene_description, detections):
    with connection.cursor() as cursor:
        # Insert into scene_logs
        cursor.execute(
            "INSERT INTO scene_logs (scene_description) VALUES (%s) RETURNING id",
            [scene_description]
        )
        scene_log_id = cursor.fetchone()[0]

        # Insert object detections
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cursor.execute(
                """
                INSERT INTO object_detections (scene_log_id, label, box_x1, box_y1, box_x2, box_y2, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                [scene_log_id, det["label"], x1, y1, x2, y2, det["confidence"]]
            )

        # Insert OCR results
        for det in detections:
            if "ocr_text" in det:
                for ocr in det["ocr_text"]:
                    cursor.execute(
                        """
                        INSERT INTO ocr_results (scene_log_id, text, confidence)
                        VALUES (%s, %s, %s)
                        """,
                        [scene_log_id, ocr["text"], ocr["confidence"]]
                    )
                    
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import SceneLog  # replace with your model
from .serializers import SceneLogSerializer  # you'll define this

@api_view(['GET'])
def get_scene_logs(request):
    logs = SceneLog.objects.order_by('-timestamp')[:5]
    serializer = SceneLogSerializer(logs, many=True)
    return Response(serializer.data)
   


