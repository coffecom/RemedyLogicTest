from django.db import models

# Create your models here.

class ImageEntity(models.Model):
    image = models.ImageField(upload_to='upload/')
