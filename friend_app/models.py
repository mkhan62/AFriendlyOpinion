from django.db import models

# Create your models here.
class Sentiment(models.Model):
    text = models.TextField(unique=True)
    category = models.CharField(max_length=20)
    intensity = models.FloatField()

    def __str__(self):
        return self.text
