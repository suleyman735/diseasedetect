from django.urls import path

from .views import Disease

app_name='disease'

urlpatterns = [
    path('',Disease.as_view(), name='myview'),
]