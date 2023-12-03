from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse
from datetime import datetime
from django.utils import formats
from django.db import connection
from django.contrib import messages
from .models import Account, Disease, Medicine, Record, Prescription, Image
import tensorflow as tf
import numpy as np
from io import BytesIO
import boto3

custombucket = 'fyp-website'
# Load the trained model (this should be done only once when the server starts)
model : tf.keras.Sequential = tf.keras.models.load_model('sure_model.h5')
oralModel : tf.keras.Sequential = tf.keras.models.load_model('88%.keras')
labels = ['Actinic Keratosis', 'Atopic Dermatitis', 'Basal Cell Carcinoma', 'Dermatofibroma', 'Eczema', 'Melanocytic Nevi', 'Melanoma', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion']
oralLabels = ['Calculus','Data Caries','Gingivitis','Hypodontia','Mouth Ulcer','Oral Cancer','Discoloration']

# Create your views here.
def login(request):
    if request.method == 'POST':
        try:
            user = Account.objects.get(phoneNo=request.POST.get('phone'),
                                    password=request.POST.get('psw'))
            
            if user:
                request.session['phone'] = user.phoneNo
                return redirect("detect")
                
        except Account.DoesNotExist:
            messages.error(request, 'The phone number or password is wrong.')
            

    return render(request, "login.html", {})

def signup(request):
    if request.method == 'POST':
        if request.POST.get('psw') == request.POST.get('cfmPsw'):
            acc = Account.objects.create(name=request.POST.get('name'), 
                                    phoneNo=request.POST.get('phone'),
                                    email=request.POST.get('email'),
                                    password=request.POST.get('psw'))
            return redirect("login")
        else:
            messages.error(request, 'The password is not matched with the confirmation password.')
    return render(request, "signup.html", {})

def detect(request):
    if request.method == 'POST':
        # Load and preprocess the input image
        uploaded_file = request.FILES['img'] 

        # preprocess the uploaded image
        img = tf.keras.preprocessing.image.load_img(BytesIO(uploaded_file.read()), target_size=(128, 128))  # Adjust the target size
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize pixel values (if not already normalized during training)

        # Make predictions
        predictions = model.predict(img)

        # Get the top N predicted class indices and their corresponding probabilities
        N = 3 
        top_N_indices = np.argsort(predictions[0])[::-1][:N]  # Get indices in descending order of probability
        top_N_probabilities = predictions[0][top_N_indices]

        # Map class indices to label names
        top_N_labels = [labels[index] for index in top_N_indices]

        # get the login user
        user = Account.objects.get(phoneNo=request.session['phone'])

        img_name = f"img_{user.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{uploaded_file.name.split('.')[-1]}"

        # # Uplaod image file in S3 #
        # s3 = boto3.resource("s3")
        # s3.Bucket(custombucket).put_object(Key=img_name, Body=uploaded_file)

        uploaded_file.name = img_name
        disease_img = Image.objects.create(path=uploaded_file)

        # Create a list of predictions and their probabilities
        diagnosis_result = []
        prescriptions = []
        for label, probability in zip(top_N_labels, top_N_probabilities):
            probability_formatted = round(float(probability), 4)

            # insert the new record
            record = Record.objects.create(patient=user, 
                                    disease=Disease.objects.filter(name=label).first(),
                                    probability=probability_formatted,
                                    disease_img=disease_img)
            # print(record.disease.pk)
            prescriptions.append(record.disease.pk)

            if(probability_formatted > 0):
                diagnosis_result.append({'disease' : Disease.objects.filter(name=label).values().first(), 'probability' : record.probability})

        # medicine_list = recommendation(diagnosis_result)
        request.session['results'] = diagnosis_result
        request.session['disease_img'] = Image.objects.filter(path=disease_img.path).values().first()
        request.session['prescriptions'] = prescriptions
        return redirect('diagnosis')    
    return render(request, "detect.html", {})     

def diagnosis(request):
    results = request.session['results']
    img = request.session['disease_img']
    return render(request, "diagnosis.html", {"results" : results, "img" : img})

def home(request):
    return render(request, "home.html")

def oralhome(request):
    return render(request, "oralHome.html")

def skinhome(request):
    return render(request, "skinHome.html")

def profile(request):
    return render(request, "profile.html")

def detectoral(request):
    if request.method == 'POST':
        # Load and preprocess the input image
        uploaded_file = request.FILES['oralImg'] 

        # preprocess the uploaded image
        img = tf.keras.preprocessing.image.load_img(BytesIO(uploaded_file.read()), target_size=(96, 96))  # Adjust the target size
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.array(img).reshape(-1, 96, 96, 3)
        img = img / 255.0  # Normalize pixel values (if not already normalized during training)

        # Make predictions
        predictions = oralModel.predict(img)

        # Get the top N predicted class indices and their corresponding probabilities
        N = 3 
        top_N_indices = np.argsort(predictions[0])[::-1][:N]  # Get indices in descending order of probability
        top_N_probabilities = predictions[0][top_N_indices]

        # Map class indices to label names
        top_N_labels = [oralLabels[index] for index in top_N_indices]

        # get the login user
        user = Account.objects.get(phoneNo=request.session['phone'])
        disease_img = Image.objects.create(img=uploaded_file)

        img_name = "img_" + str(user.name) + "_" + formats.date_format(datetime.now(), "DATETIME_FORMAT")
        # # Uplaod image file in S3 #
        # uploaded_file.name = img_name
        # s3 = boto3.resource("s3")
        # s3.Bucket(custombucket).put_object(Key=img_name, Body=uploaded_file)
        # # disease_img = Image.objects.create(path=uploaded_file)

        # Create a list of predictions and their probabilities
        diagnosis_result = []
        for label, probability in zip(top_N_labels, top_N_probabilities):
            probability_formatted = round(float(probability), 4)

            # insert the new record
            record = Record.objects.create(patient=user, 
                                    disease=Disease.objects.filter(name=label).first(),
                                    probability=probability_formatted,
                                    disease_img=disease_img)
            
            if(probability_formatted > 0):
                diagnosis_result.append(Disease.objects.filter(name=label).values().first())
        # Print or return the list of predictions
        # text = ""
        # for index, prediction in enumerate(diagnosis_result):
        #     text = text + f"{index + 1}. {prediction['disease'].cause}, Probability: {prediction['probability']:.4f}"
        # return HttpResponse(text)
        print(disease_img.img)
        print(Image.objects.filter(img=disease_img.img).values().first())
        request.session['results'] = diagnosis_result
        request.session['disease_img'] = Image.objects.filter(img=disease_img.img).values().first()
        return redirect('diagnosisoral')    
    return render(request, "detectOral.html", {})

def diagnosisoral(request):
    results = request.session['results']
    img = request.session['disease_img']
    return render(request, "diagnosisOral.html", {"results" : results, "img" : img})