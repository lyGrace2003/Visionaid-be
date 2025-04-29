# api/urls.py
from django.urls import path
from .views import upload_image
# ocr_view

urlpatterns = [
    path('upload-image/', upload_image, name='upload_image'),  # Adjusted path for image upload
    #path('get-ocr-result/', get_ocr_result, name="ocr-result")
]
