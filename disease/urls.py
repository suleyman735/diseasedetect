from django.urls import path

from .views import PredictView

app_name='disease'

urlpatterns = [
    path('predict/',PredictView.as_view(), name='myview'),
]