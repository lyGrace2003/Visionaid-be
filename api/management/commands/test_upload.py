from django.core.management.base import BaseCommand
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from pathlib import Path

class Command(BaseCommand):
    help = "Test upload_image view with a static image."

    def handle(self, *args, **kwargs):
        client = Client()
        image_path = Path(r"C:\Users\User\OneDrive\Pictures\Camera Roll\WIN_20250424_01_28_08_Pro.jpg")  

        if not image_path.exists():
            self.stdout.write(self.style.ERROR(f"Image not found: {image_path}"))
            return

        with open(image_path, 'rb') as img:
            image_file = SimpleUploadedFile(img.name, img.read(), content_type='image/jpeg')
            response = client.post('/api/upload-image/', {'image': image_file})

        self.stdout.write(self.style.SUCCESS("Response:"))
        self.stdout.write(response.content.decode())
