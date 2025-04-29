# visionaid/urls.py
from django.contrib import admin
from django.urls import path, include  # Import include to include the API URLs

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
] 
