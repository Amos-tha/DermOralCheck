from django.urls import path
from dermoral import views

urlpatterns = [
    path('', views.login, name="login"),
    path('signup', views.signup, name="signup"),
    path('detect', views.detect, name="detect"),
    path('diagnosis', views.diagnosis, name="diagnosis"),
    path('home', views.home, name="home"),
    path('skinhome', views.skinhome, name='skinhome'),
    path('oralhome', views.oralhome, name='oralhome'),
    path('map', views.map, name="map"),
    path('profile', views.profile, name='profile'),
    path('detectoral', views.detectoral, name='detectoral'),
    path('diagnosisoral', views.diagnosisoral, name="diagnosisoral"),
    path('camera', views.live_cam, name='livecamera'),
]