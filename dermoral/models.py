from django.db import models

# Create your models here.
class Account(models.Model):
    phoneNo = models.CharField(primary_key=True, unique=True, max_length=20)
    name = models.CharField(max_length=250)
    email = models.EmailField(max_length=255)
    address = models.CharField(max_length=250, blank=True)
    password = models.CharField(max_length=250)
    healthCondition = models.CharField(max_length=250, blank=True)

class Disease(models.Model):
    # default auto increment if no put this line of code
    # id = models.AutoField(primary_key=True) 
    name = models.CharField(max_length=250)
    description = models.CharField(max_length=250)
    symptom = models.CharField(max_length=250)
    cause = models.CharField(max_length=250)

class Medicine(models.Model):
    name = models.CharField(max_length=250)
    description = models.CharField(max_length=250)
    sideEffect = models.CharField(max_length=250)

class Record(models.Model):
    patient = models.ForeignKey(Account, on_delete=models.CASCADE,)
    disease = models.ForeignKey(Disease, on_delete=models.CASCADE,)
    recordDate = models.DateField(auto_now=True)
    recordTime = models.TimeField(auto_now=True)
    disease_img = models.ImageField(upload_to='static/media')

class Prescription(models.Model):
    disease = models.ForeignKey(Disease, on_delete=models.CASCADE,)
    medicine = models.ForeignKey(Medicine, on_delete=models.CASCADE,)







