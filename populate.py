import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AFriendsOpinion.settings')
import django
django.setup()
import data_miner as dm
from friend_app.models import Sentiment
import pandas as pd

def add_sentiment(line):
    text = line.text
    category = line.sentiment
    intensity = line.intensity
    model = Sentiment(text=text,category=category,intensity=intensity)
    model.save()

if __name__ == '__main__':
    print("---Populating Database---")
    data = dm.combined_frame
    data.apply(add_sentiment, axis=1)
    print("---Population Successful---")
