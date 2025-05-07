from rest_framework import serializers
from .models import SceneLog

class SceneLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = SceneLog
        fields = ['id', 'description', 'timestamp']
