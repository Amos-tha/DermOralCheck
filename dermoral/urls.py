from django.urls import path
from dermoral import views

urlpatterns = [
    path('', views.home, name="home"),
]