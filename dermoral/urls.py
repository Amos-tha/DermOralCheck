from django.urls import path
from dermoral import views

urlpatterns = [
    path("skin/test", views.home, name="home"),
]