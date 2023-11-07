from django.shortcuts import render
from django.http import HttpResponse
from django.db import connection
from .models import Account, Disease, Medicine, Record, Prescription

# Create your views here.
def home(request):
    return render(request, "login.html", {})

def signup(request):
    if request.method == 'POST':
        acc = Account.objects.create(name=request.POST.get('name'), 
                                   phoneNo=request.POST.get('phone'),
                                   email=request.POST.get('email'),
                                   password=request.POST.get('psw'))
        return HttpResponse(acc.phoneNo)
    return render(request, "signup.html", {})

# def register_acc(request):
    
        


def db_table_exists():
    # tables = db_table_exists()
    # return HttpResponse(tables)
    tables = connection.introspection.table_names()
    return tables 
