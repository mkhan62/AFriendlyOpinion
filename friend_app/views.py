from django.shortcuts import render
from friend_app.models import Sentiment
from friend_app.forms import SentimentForm
import pandas as pd
from predictor import Predictor

def dataset_access():
    data = list(Sentiment.objects.all())
    data_dict = {'text': [], 'category': [], 'intensity': []}
    for i in range(len(data)):
        data_dict['text'].append(data[i].text)
        data_dict['category'].append(data[i].category)
        data_dict['intensity'].append(data[i].intensity)
    df = pd.DataFrame(data_dict)
    X = df['text']
    y = df['intensity'].apply(lambda x: round(x,1))
    z = df['category']
    return (X, y, z)

def index(request):
    form = SentimentForm()
    if request.method == 'POST':
        form = SentimentForm(request.POST)

        if form.is_valid():
            form.save(commit=False)
            text = form.cleaned_data['text']
            X, y, z = dataset_access()
            model_category = Predictor(X=X,y=z)
            predicted_category = model_category.predict_category(text=[text])
            model_instensity = Predictor(X=X, y=y)
            predicted_intensity = model_instensity.predict_intensity(text=[text])
            print("The predicted category is: {}".format(predicted_category))
            print("predicted intensity is: {}".format(predicted_intensity))

    return render(request, 'friend_app/base.html', {'form': form})
