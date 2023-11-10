from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.db import connection
from django.contrib import messages
from .models import Account, Disease, Medicine, Record, Prescription

import tensorflow as tf
import numpy as np
from io import BytesIO

# Load the trained model (this should be done only once when the server starts)
model : tf.keras.Sequential = tf.keras.models.load_model('sure_model.h5')
labels = ['Actinic Keratosis', 'Atopic Dermatitis', 'Basal Cell Carcinoma', 'Dermatofibroma', 'Eczema', 'Melanocytic Nevi', 'Melanoma', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion']

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
            return render(request, 'login.html', {})
        else:
            messages.error(request, 'The password is not matched with the confirmation password.')
    return render(request, "signup.html", {})

def detect(request):
    if request.method == 'POST':
        # Load and preprocess the input image
        uploaded_file = request.FILES['img'] 
        # fs = FileSystemStorage(location="/static/media/")
        # fs.save(uploaded_file.name, uploaded_file)

        # preprocess the uploaded image
        img = tf.keras.preprocessing.image.load_img(BytesIO(uploaded_file.read()), target_size=(128, 128))  # Adjust the target size
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize pixel values (if not already normalized during training)

        # Make predictions
        predictions = model.predict(img)

        # Get the top N predicted class indices and their corresponding probabilities
        N = 3  # Adjust as needed
        top_N_indices = np.argsort(predictions[0])[::-1][:N]  # Get indices in descending order of probability
        top_N_probabilities = predictions[0][top_N_indices]

        # Map class indices to label names
        top_N_labels = [labels[index] for index in top_N_indices]

        # get the login user
        user = Account.objects.get(phoneNo=request.session['phone'])

        # Create a list of predictions and their probabilities
        diagnosis_result = []
        for label, probability in zip(top_N_labels, top_N_probabilities):
            probability_formatted = round(probability, 4)

            # get the related disease object based on the name
            disease = Disease.objects.get(name=label) 

            # insert the new record
            record = Record.objects.create(patient=user, 
                                    disease=disease,
                                    disease_img=uploaded_file)
            
            if(probability_formatted > 0):
                diagnosis_result.append({"disease": disease, "probability": probability_formatted})
        # Print or return the list of predictions
        text = ""
        for index, prediction in enumerate(diagnosis_result):
            text = text + f"{index + 1}. {prediction['disease'].cause}, Probability: {prediction['probability']:.4f}"
        # return HttpResponse(text)
        return redirect("diagnosis", result = diagnosis_result)    
    return render(request, "detect.html", {})    

def diagnosis(request):
    result = request.GET.get("result")
    return render(request, "diagnosis.html", results=result)
