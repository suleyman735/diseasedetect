from django.db import models

# Create your models here.


class ImageUpload(models.Model):
    image = models.ImageField(upload_to='images/')
    predicted_class = models.CharField(max_length=255, null=True, blank=True)
    confidenceforpre = models.FloatField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    

