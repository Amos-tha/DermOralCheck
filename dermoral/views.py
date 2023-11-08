from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.db import connection
from django.contrib import messages
from .models import Account, Disease, Medicine, Record, Prescription

# Create your views here.
def login(request):
    if request.method == 'POST':
        try:
            user = Account.objects.get(phoneNo=request.POST.get('phone'),
                                    password=request.POST.get('psw'))
            if user:
                return redirect("home")
                
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
            print(acc)
            return render(request, 'login.html', {})
        else:
            messages.error(request, 'The password is not matched with the confirmation password.')
    return render(request, "signup.html", {})

def home(request):
    return render(request, "home.html", {})    

# def db_table_exists():
#     # tables = db_table_exists()
#     # return HttpResponse(tables)
#     tables = connection.introspection.table_names()
#     return tables 
