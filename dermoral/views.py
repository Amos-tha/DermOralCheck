from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse
from datetime import datetime
from django.db import connection
from django.contrib import messages
from .models import Account, Disease, Medicine, Record, Prescription, Image
import tensorflow as tf
import numpy as np
from io import BytesIO
import pandas as pd
import boto3

custombucket = 'fyp-website'
# Load the trained model (this should be done only once when the server starts)
model : tf.keras.Sequential = tf.keras.models.load_model('sure_model.h5')
labels = ['Actinic Keratosis', 'Atopic Dermatitis', 'Basal Cell Carcinoma', 'Dermatofibroma', 'Eczema', 'Melanocytic Nevi', 'Melanoma', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion']

# Create your views here.
def login(request):
    if request.method == 'POST':
        try:
            user = Account.objects.filter(phoneNo=request.POST.get('phone'),
                                    password=request.POST.get('psw')).first()
            
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

        medicine_list = recommendation(diagnosis_result)
        request.session['results'] = diagnosis_result
        request.session['disease_img'] = Image.objects.filter(path=disease_img.path).values().first()
        request.session['prescriptions'] = prescriptions
        return redirect('diagnosis')    
    return render(request, "detect.html", {})   

def recommendation(results):
    medicine_list = []
    for disease in results:
        # print(disease['disease']['id'])
        mid = Prescription.objects.filter(disease=disease['disease']['id']).values('medicine_id').first()
        if mid:
            medicine_list.append(Medicine.objects.filter(pk=mid['medicine_id']).values().first())

    return medicine_list
    

# def recommendation(request):
#     # Read patient preferences and health conditions from Excel
#     patient_data = pd.read_excel("patient_data.xlsx")

#     # Fetch user ID or any identifier from the request
#     user_id = request.GET.get('user_id', 'default_user')

#     # Get user preferences and health condition from the Excel data
#     user_preferences = patient_data.loc[patient_data['user_id'] == user_id, 'preference'].values[0]
#     user_health_condition = patient_data.loc[patient_data['user_id'] == user_id, 'health_condition'].values[0]

#     # Retrieve medicines data from the database
#     medicines = Medicine.objects.filter(health_condition=user_health_condition)

#     # Convert medicines data to a Pandas DataFrame
#     medicines_df = pd.DataFrame(list(medicines.values()))

#     # Filter medicines based on user preference
#     filtered_medicines = medicines_df[medicines_df['preference'] == user_preferences]

#     # Calculate components for the weighted average
#     v = filtered_medicines['feedback_count']
#     R = filtered_medicines['average_rating']
#     C = R.mean()
#     m = v.quantile(0.70)

#     # Calculate the weighted average
#     filtered_medicines['weighted_average'] = ((R * v) + (C * m)) / (v + m)

#     # Sort medicines based on weighted average and average rating
#     sorted_medicines = filtered_medicines.sort_values(['weighted_average', 'average_rating'], ascending=[False, False])

#     # Display the top recommended medicines
#     top_medicines = sorted_medicines[['name', 'feedback_count', 'average_rating', 'weighted_average', 'preference']].head(20)

#     # Pass the recommended medicines and user information to the template
#     context = {'top_medicines': top_medicines, 'user_preferences': user_preferences, 'user_health_condition': user_health_condition}

#     return render(request, 'recommendations.html', context) 

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

def map(request):
    return render(request, "map.html")

