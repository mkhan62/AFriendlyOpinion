from django.forms import ModelForm
from friend_app import models

class SentimentForm(ModelForm):
    class Meta:
        model = models.Sentiment
        fields = ('text',)
