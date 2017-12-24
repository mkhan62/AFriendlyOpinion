from django.shortcuts import render
from friend_app.models import Sentiment
from friend_app.forms import SentimentForm
from data_miner import preprocess, pipeline
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from predictor import Predictor



data = list(Sentiment.objects.all())
data_dict = {'text': [], 'category': [], 'intensity': []}
for i in range(len(data)):
    data_dict['text'].append(data[i].text)
    data_dict['category'].append(data[i].category)
    data_dict['intensity'].append(data[i].intensity)
df = pd.DataFrame(data_dict)
X = df['text']
y = df['intensity'].apply(lambda x: round(x,1))
model = pipeline.fit(X, y)

def index(request):
    form = SentimentForm()
    if request.method == 'POST':
        form = SentimentForm(request.POST)

        if form.is_valid():
            form.save(commit=False)
            text = form.cleaned_data['text']
            prediction = Predictor.predict_intensity(text=text)
            print(prediction)

    return render(request, 'friend_app/base.html', {'form': form})
