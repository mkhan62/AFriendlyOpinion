from django.conf.urls import url
from friend_app import views


app_name= 'friend_app'

urlpatterns = [
    url(r'^entry_form/$', views.entry_form, name='entry_form'),
    url(r'^loader/$', views.loader, name='loader')
]
