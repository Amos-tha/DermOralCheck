from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import StreamingHttpResponse, HttpResponseRedirect
from datetime import datetime
from django.db import connection
from django.contrib import messages
from .models import Account, Disease, Medicine, Record, Prescription, Image
from django.forms.models import model_to_dict
import tensorflow as tf
import numpy as np
from io import BytesIO
import pandas as pd
import cv2
import threading
import boto3
from django.views.decorators import gzip

custombucket = 'fyp-website'
# Load the trained model (this should be done only once when the server starts)
model : tf.keras.Sequential = tf.keras.models.load_model('sure_model.h5')
oralModel : tf.keras.Sequential = tf.keras.models.load_model('88%.keras')
labels = ['Actinic Keratosis', 'Atopic Dermatitis', 'Basal Cell Carcinoma', 'Dermatofibroma', 'Eczema', 'Melanocytic Nevi', 'Melanoma', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion']
oralLabels = ['Calculus','Dental Caries','Gingivitis','Hypodontia','Mouth Ulcer','Oral Cancer','Tooth Discoloration']

def login(request):
    if request.method == 'POST':
            user = Account.objects.filter(phoneNo=request.POST.get('phone'),
                                    password=request.POST.get('psw')).first()
            if user:
                request.session['phone'] = user.phoneNo
                return redirect("home")
            else:
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
    if request.method == 'POST' or request.GET.get('method') == 'POST':
        user = Account.objects.get(phoneNo=request.session['phone'])
        if 'img' in request.FILES:
            # Load and preprocess the input image
            uploaded_file = request.FILES['img'] 
            img = tf.keras.preprocessing.image.load_img(BytesIO(uploaded_file.read()), target_size=(128, 128))  # Adjust the target size
            # get the login user

            img_name = f"img_{user.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{uploaded_file.name.split('.')[-1]}"

            # # Uplaod image file in S3 #
            # s3 = boto3.resource("s3")
            # s3.Bucket(custombucket).put_object(Key=img_name, Body=uploaded_file)
            uploaded_file.name = img_name
            disease_img = Image.objects.create(path=uploaded_file)
            
        else:
            uploaded_file = request.session['save_path']
            saved_img = cv2.imread(uploaded_file)
            img = cv2.resize(saved_img, (128,128))
            disease_img = Image.objects.create(path=uploaded_file)


        # preprocess the uploaded image
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize pixel values (if not already normalized during training)

        # Make predictions
        predictions = model.predict(img)
        print(predictions)

        # Get the top N predicted class indices and their corresponding probabilities
        N = 3 
        top_N_indices = np.argsort(predictions[0])[::-1][:N]  # Get indices in descending order of probability
        top_N_probabilities = predictions[0][top_N_indices]

        # Map class indices to label names
        top_N_labels = [labels[index] for index in top_N_indices]

        # Create a list of predictions and their probabilities
        diagnosis_result = []
        medicine_list = []

        for label, probability in zip(top_N_labels, top_N_probabilities):
            probability_formatted = round(float(probability), 4)
            if(probability_formatted > 0):
                disease = Disease.objects.filter(name=label).first()
                dict_disease = model_to_dict(disease)

                # print(dict_disease)
                # print(dict_disease['name'])
                print("test")
                # insert the new record
                record = Record.objects.create(patient=user, 
                                        disease=disease,
                                        probability=probability_formatted,
                                        disease_img=disease_img)
                # print(record.disease.pk)
                mids = Prescription.objects.filter(disease=disease).values_list('medicine_id', flat=True)

                if mids.exists():
                    for mid in mids:
                        medicine_list.append(Medicine.objects.filter(pk=mid).values().first())
                        print(mid)
                else:
                    medicine_list = []

                diagnosis_result.append({'disease' : dict_disease, 'probability' : record.probability, 'medicines' : medicine_list})
                
                # # check the medicine found?
                # if mid:
                #     # If it's not, create a new list with the current medicine data as the first element
                #     medicine_list[dict_disease['name']] = [Medicine.objects.filter(pk=mid['medicine_id']).values_list()]

        request.session['results'] = diagnosis_result
        request.session['disease_img'] = Image.objects.filter(path=disease_img.path).values().first()
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

# read camera
global_camera = None

def skinhome(request):
    global global_camera
    if(request.GET.get("freeze")):
        global_camera.freeze_frame()
    elif(request.GET.get("unfreeze")):
            global_camera.unfreeze_frame()
    elif(request.GET.get("release")):
        global_camera.release_camera()
        
    return render(request, "skinHome.html")

def map(request):
    return render(request, "map.html")

def profile(request):
    return render(request, "profile.html")

def detectoral(request):
    if request.method == 'POST':
        # Load and preprocess the input image
        uploaded_file = request.FILES['img'] 

        # preprocess the uploaded image
        img = tf.keras.preprocessing.image.load_img(BytesIO(uploaded_file.read()), target_size=(96, 96))  # Adjust the target size
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.array(img).reshape(-1, 96, 96, 3)
        img = img / 255.0  # Normalize pixel values (if not already normalized during training)

       # Make predictions
        predictions = oralModel.predict(img)
        print(predictions)

        # Get the top N predicted class indices and their corresponding probabilities
        N = 3 
        top_N_indices = np.argsort(predictions[0])[::-1][:N]  # Get indices in descending order of probability
        top_N_probabilities = predictions[0][top_N_indices]

        # Map class indices to label names
        top_N_labels = [oralLabels[index] for index in top_N_indices]

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
        medicine_list = []

        for label, probability in zip(top_N_labels, top_N_probabilities):
            probability_formatted = round(float(probability), 4)
            if(probability_formatted > 0):
                disease = Disease.objects.filter(name=label).first()
                dict_disease = model_to_dict(disease)

                # print(dict_disease)
                # print(dict_disease['name'])
                print("test")
                # insert the new record
                record = Record.objects.create(patient=user, 
                                        disease=disease,
                                        probability=probability_formatted,
                                        disease_img=disease_img)
                # print(record.disease.pk)
                mids = Prescription.objects.filter(disease=disease).values_list('medicine_id', flat=True)

                if mids.exists():
                    for mid in mids:
                        medicine_list.append(Medicine.objects.filter(pk=mid).values().first())
                        print(mid)
                else:
                    medicine_list = []

                diagnosis_result.append({'disease' : dict_disease, 'probability' : record.probability, 'medicines' : medicine_list})
                
                # # check the medicine found?
                # if mid:
                #     # If it's not, create a new list with the current medicine data as the first element
                #     medicine_list[dict_disease['name']] = [Medicine.objects.filter(pk=mid['medicine_id']).values_list()]

        request.session['results'] = diagnosis_result
        request.session['disease_img'] = Image.objects.filter(path=disease_img.path).values().first()
        return redirect('diagnosisoral')    
    return render(request, "detectOral.html", {})

def diagnosisoral(request):
    results = request.session['results']
    img = request.session['disease_img']
    return render(request, "diagnosisOral.html", {"results" : results, "img" : img})
    # return render(request, "diagnosisOral.html", {"results" : results, "img" : img})

# read camera
# global_camera = None
run_camera = False
class VideoCamera(object):
    def __init__(self) :
        self.video = None
        self.frame = None
        self.grabbed = False
        self.frozen_frame = None
        self.freeze_flag = threading.Event()
        self.initialize_video_capture()

    def initialize_video_capture(self):
        global run_camera
        run_camera = True
        self.video = cv2.VideoCapture(0)
        _, self.frame = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        # self.video.release()
        pass
 
    def get_frame(self):
        if self.frozen_frame is not None:
            _, jpeg = cv2.imencode('.jpg', self.frozen_frame)
            return jpeg.tobytes()
        else:
            image = self.frame
            _, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()

    def update(self):
        while True:
            if not self.freeze_flag.is_set():
                (self.grabbed, self.frame) = self.video.read()

    def freeze_frame(self):
        if not self.freeze_flag.is_set():
            self.frozen_frame = self.frame
            self.freeze_flag.set()

    def unfreeze_frame(self):
        if self.freeze_flag.is_set():
            self.frozen_frame = None
            self.freeze_flag.clear()

    def capture_and_save_frame(self, save_path):
        if self.frozen_frame is not None:
            cv2.imwrite(save_path, self.frozen_frame)
            self.frozen_frame = None
            self.freeze_flag.clear()

    def release_camera(self):
        global run_camera
        run_camera = False
        self.frozen_frame = None
        self.frame = None
        self.freeze_flag.clear()
        self.video.release()

def cam(camera):
    global run_camera
    while run_camera:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\n' + frame + b'\r\n\r\n')

@gzip.gzip_page
def live_cam(request):
    try:
        global global_camera 
        global_camera = VideoCamera()
        return StreamingHttpResponse(cam(global_camera), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        print(f"Error: {e}")

def save_frame(request):
    user = Account.objects.get(phoneNo=request.session['phone'])
    img_name = f"img_{user.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    save_path = 'static/media/' + img_name
    global_camera.capture_and_save_frame(save_path)
    request.session["save_path"] = save_path
    return HttpResponseRedirect(reverse('detect') + '?method=POST')

