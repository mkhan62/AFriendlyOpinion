from django.shortcuts import render
from friend_app.models import Sentiment
from friend_app.forms import SentimentForm

# Create your views here.
def index(request):
    form = SentimentForm()
    if request.method == 'POST':
        form = SentimentForm(request.POST)

        if form.is_valid():
            form.save(commit=False)
            text = form.cleaned_data['text']
            print(text)

    return render(request, 'friend_app/base.html', {'form': form})
