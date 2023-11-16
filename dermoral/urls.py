from django.urls import path
from dermoral import views

urlpatterns = [
    path('', views.login, name="login"),
    path('signup', views.signup, name="signup"),
    path('detect', views.detect, name="detect"),
    path('diagnosis', views.diagnosis, name="diagnosis"),
    path('home', views.home, name="home") #need do a view file for this
]