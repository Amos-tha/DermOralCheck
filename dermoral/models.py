from django.db import models

# Create your models here.
class Account(models.Model):
    phoneNo = models.CharField(primary_key=True, unique=True, max_length=20)
    name = models.CharField(max_length=200)
    email = models.CharField(max_length=200, null=True)
    address = models.CharField(max_length=200)
    password = models.CharField(max_length=200)
    healthCondition = models.CharField(max_length=200, blank=True)

class SkinDisease(models.Model):
    name = models.CharField(max_length=200)
    description = models.CharField(max_length=200)
    symptom = models.CharField(max_length=200)
    cause = models.CharField(max_length=200)

class Record(models.Model):
    patient = models.ForeignKey(Account, on_delete=models.CASCADE,)
    disease = models.ForeignKey(SkinDisease, on_delete=models.CASCADE,)
    disease_img = models.ImageField(upload_to='')



