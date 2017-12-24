from django.shortcuts import render
from friend_app.models import Sentiment
from friend_app.forms import SentimentForm
import pandas as pd
import pickle
import os

if os.path.getsize('cmodel.pkl') > 0:
    category_model_pkl = open('cmodel.pkl', 'rb')
    category_model = pickle.load(category_model_pkl)
    print(category_model)
if os.path.getsize('imodel.pkl') > 0:
    intensity_model_pkl = open('imodel.pkl', 'rb')
    intensity_model = pickle.load(intensity_model_pkl)


def index(request):
    return render(request, 'friend_app/index.html')

def loader(request):
    return render(request, 'friend_app/loader.html')

def entry_form(request):
    form = SentimentForm()
    if request.method == 'POST':
        form = SentimentForm(request.POST)

        if form.is_valid():
            form.save(commit=False)
            text = form.cleaned_data['text']
            # X, y, z = dataset_access()
            # model_category = Predictor(X=X,y=z)
            # predicted_category = model_category.predict_category(text=[text])
            # model_instensity = Predictor(X=X, y=y)
            # predicted_intensity = model_instensity.predict_intensity(text=[text])
            sentiment = category_model.predict([text])
            intensity = intensity_model.predict([text])
            
            print("The predicted category is: {}".format(sentiment))
            print("predicted intensity is: {}".format(intensity))

    return render(request, 'friend_app/entry_form.html', {'form': form})
