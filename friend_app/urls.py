from django.conf.urls import url
from friend_app import views


app_name= 'friend_app'

urlpatterns = [
    url(r'^entry_form/$', views.entry_form, name='entry_form'),
    url(r'^output_result/$', views.OutputView.as_view(), name='output_result'),
    url(r'^thank/$', views.thankyou, name='thank'),
]
