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
from datetime import datetime, timedelta
import pytz
from rest_framework.decorators import api_view
from rest_framework.response import Response
from api.models import SceneLog  # replace with your model
from api.serializers import SceneLogSerializer  # you'll define this

from django.conf import settings

ph_time = datetime.now(pytz.timezone("Asia/Manila"))

model_path = r"C:\Users\User\OneDrive\Desktop\thesis\VisionAid\object_detection\yolov8s.pt"
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

def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]) 
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def bbox_from_polygon(polygon):
    xs = [p[0] for p in polygon]  #extract x coordinates
    ys = [p[1] for p in polygon]  # extract y coordinates
    return [min(xs), min(ys), max(xs), max(ys)]

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

        if not detections:
            detections = [] 

        # Get all detected object boxes
        object_boxes = [d["box"] for d in detections]
        
        #Perform full image OCR
        full_ocr_results = reader.readtext(original_img)

        # Filter out OCR texts that overlap with detected objects
        filtered_full_ocr = []
        for (bbox, text, conf) in full_ocr_results:
            if conf < 0.5: #skip if confidence less than 50% 
                continue
            ocr_box = bbox_from_polygon(bbox) #convert polygon to bounding box 
            overlaps = any(box_iou(ocr_box, det_box) > 0.3 for det_box in object_boxes) 
            if not overlaps:
                filtered_full_ocr.append({
                    "box": list(map(int, ocr_box)),
                    "text": text,
                    "confidence": float(conf)
                })

        print("Full OCR (outside object regions):", filtered_full_ocr)

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

            # Draw bounding boxes for OCR results within detected objects (white color)
            for ocr in detection["ocr_text"]:
                ocr_box = [x1, y1, x2, y2]  # Use the object's bounding box as a reference
                cv2.rectangle(original_img, (ocr_box[0], ocr_box[1]), (ocr_box[2], ocr_box[3]), (255, 255, 255), 2)
            
        # Draw bounding boxes for full-page OCR results (blue color)
        for ocr_entry in filtered_full_ocr:
            ocr_box = ocr_entry["box"]
            cv2.rectangle(original_img, (ocr_box[0], ocr_box[1]), (ocr_box[2], ocr_box[3]), (255, 0, 0), 2)

        aggregated_results = []
        
        for det in detections:
            aggregated_results.append({
                "label": det["label"],
                "box": det["box"],
                "confidence": det["confidence"],
                "ocr_text": det.get("ocr_text", [])
            })
        
        # Add full-page OCR results (outside object boxes)
        for ocr_entry in filtered_full_ocr:
            aggregated_results.append({
                "label": "ocr",
                "box": ocr_entry["box"],
                "confidence": ocr_entry["confidence"],
                "ocr_text": [ { "text": ocr_entry["text"], "confidence": ocr_entry["confidence"] } ]
            })
        
        print("aggregated results", aggregated_results)

        # Ensure that bounding box coordinates are integers
        for result in aggregated_results:
            result['box'] = list(map(int, result['box']))  # Explicitly cast to int
            # If there are any other NumPy-specific types, ensure they are converted too
            if isinstance(result['confidence'], np.int32) or isinstance(result['confidence'], np.float32):
                result['confidence'] = float(result['confidence'])  # Convert confidence to float if necessary


        try:
            scene_description = generate_scene_description(aggregated_results)
        except Exception as e:
            print("Scene generation error:", e)
            scene_description = None  # Ensure it's set to None in case of an error

        if scene_description:
            try:
                save_scene_description_to_db(scene_description, aggregated_results)
            except Exception as e:
                print("Error saving to DB:", e)
                return JsonResponse({"error": "Failed to save to database."}, status=500)
        else:
            return JsonResponse({"error": "Scene description generation failed."}, status=500)



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

        if conf < 0.4:  # skip if confidence less than 40%
            continue

        class_id = int(class_id)
        label = class_names[class_id] if class_id < len(class_names) else str(class_id)  #convert classid from detection to labels
        #If class_id is out of range (very rare), it just returns the ID as a string.

        if len([x1, y1, x2, y2]) == 4:
            box_x1, box_y1, box_x2, box_y2 = int(x1), int(y1), int(x2), int(y2)
        else:
            # Default or handle missing bounding box data
            box_x1, box_y1, box_x2, box_y2 = 0, 0, 0, 0  # Defaults if invalid/missing

        raw_detections.append({
            "label": label,
            "box": [box_x1, box_y1, box_x2, box_y2],
            "confidence": round(conf, 2)
        })

    return raw_detections

from django.db import connection

import json

def save_scene_description_to_db(scene_description, detections):
    with connection.cursor() as cursor:
        # Insert into scene_logs
        cursor.execute(
            "INSERT INTO scene_logs (scene_description, created_at) VALUES (%s, %s) RETURNING id",
            [scene_description, ph_time]
        )
        scene_log_id = cursor.fetchone()[0]

        for det in detections:
            x1, y1, x2, y2 = det["box"]

            if det["label"] == "ocr":
                # Skip object detection and save to ocr_results directly
                for ocr in det.get("ocr_text", []):
                    cursor.execute(
                        """
                        INSERT INTO ocr_results (scene_log_id, box_x1, box_y1, box_x2, box_y2, text, confidence)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        [scene_log_id, x1, y1, x2, y2, ocr["text"], ocr["confidence"]]
                    )
                continue

            # Otherwise, save to object_detections + ocr_results for object-related OCR
            cursor.execute(
                """
                INSERT INTO object_detections (scene_log_id, label, box_x1, box_y1, box_x2, box_y2, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                [scene_log_id, det["label"], x1, y1, x2, y2, det["confidence"]]
            )
            for ocr in det.get("ocr_text", []):
                cursor.execute(
                    """
                    INSERT INTO ocr_results (scene_log_id, box_x1, box_y1, box_x2, box_y2, text, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    [scene_log_id, x1, y1, x2, y2, ocr["text"], ocr["confidence"]]
                )
                    

@api_view(['GET'])
def get_scene_logs(request):
    five_days_ago = ph_time - timedelta(days=5)

    logs = SceneLog.objects.filter(created_at__gte=five_days_ago, created_at__lte=ph_time).order_by('-created_at')

    serializer = SceneLogSerializer(logs, many=True)
    return Response(serializer.data)
   


