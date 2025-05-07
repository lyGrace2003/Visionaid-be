# api/urls.py
from django.urls import path
from .views import upload_image
from .views import get_scene_logs
# ocr_view

urlpatterns = [
    path('upload-image/', upload_image, name='upload_image'),  # Adjusted path for image upload
    #path('get-ocr-result/', get_ocr_result, name="ocr-result")
    path('scene-logs/', get_scene_logs),
]
