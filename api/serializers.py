from rest_framework import serializers
from .models import SceneLog

class SceneLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = SceneLog
        fields = ['id', 'scene_description', 'created_at']
