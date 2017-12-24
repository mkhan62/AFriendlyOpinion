from django.shortcuts import render, redirect
from friend_app.models import Sentiment
from friend_app.forms import SentimentForm
from django.views.generic import View, TemplateView, DetailView
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


class IndexView(TemplateView):
    template_name = 'friend_app/index.html'


class OutputView(TemplateView):
    template_name = 'friend_app/output_result.html'


def entry_form(request):
    form = SentimentForm()
    if request.method == 'POST':
        form = SentimentForm(request.POST)

        if form.is_valid():
            form.save(commit=False)
            text = form.cleaned_data['text']
            sentiment = category_model.predict([text])
            intensity = intensity_model.predict([text])
            print("The predicted category is: {}".format(sentiment[0]))
            print("predicted intensity is: {}".format(intensity[0]))
            return redirect('friend_app:output_result',)

    return render(request, 'friend_app/entry_form.html', {'form': form})
