import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AFriendsOpinion.settings')
import django
django.setup()
from friend_app.models import Sentiment
import pandas as pd

def frame_gen(data1, data2, data3):
    col = ['id', 'text', 'sentiment', 'intensity']
    data1_frame = pd.read_csv(data1, sep='\t', names=col)
    data2_frame = pd.read_csv(data2, sep='\t', names=col)
    data3_frame = pd.read_csv(data3, sep='\t', names=col)
    frames = [data1_frame, data2_frame, data3_frame]
    final_frame = pd.concat(frames)
    return final_frame


def add_sentiment(line):
    text = line.text
    category = line.sentiment
    intensity = line.intensity
    model = Sentiment(text=text,category=category,intensity=intensity)
    model.save()


anger_data = frame_gen('datasets/anger1.txt', 'datasets/anger2.txt', 'datasets/anger3.txt')
fear_data = frame_gen('datasets/fear1.txt', 'datasets/fear2.txt', 'datasets/fear3.txt')
joy_data = frame_gen('datasets/joy1.txt', 'datasets/joy2.txt', 'datasets/joy3.txt')
sadness_data = frame_gen('datasets/sadness1.txt', 'datasets/sadness2.txt', 'datasets/sadness3.txt')

all_list = [anger_data, fear_data, joy_data, sadness_data]
combined_frame = pd.concat(all_list)

if __name__ == '__main__':
    print("---Populating Database---")
    data = combined_frame
    data.apply(add_sentiment, axis=1)
    print("---Population Successful---")
