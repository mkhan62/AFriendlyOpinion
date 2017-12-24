from django.db import models

# Create your models here.
class Sentiment(models.Model):
    text = models.TextField()
    category = models.CharField(max_length=20)
    intensity = models.FloatField()

    def __str__(self):
        # return ('%s\t%s\t%s' % (self.text, self.category, self.intensity))
        return ('%s'%({
        'text': self.text, 'category': self.category, 'intensity': self.intensity
        }))
