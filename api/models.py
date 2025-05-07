from django.db import models

class SceneLog(models.Model):
    id = models.AutoField(primary_key=True) 
    scene_description = models.TextField()
    created_at = models.DateTimeField()

    class Meta:
        db_table = 'scene_logs'  #must match db
        managed = False

