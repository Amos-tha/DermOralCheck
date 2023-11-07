from django.shortcuts import render
from django.http import HttpResponse
from django.db import connection

# Create your views here.
def home(request):
    tables = db_table_exists()
    return HttpResponse(tables)
    return render(request, "map.html", {})

def db_table_exists():
    tables = connection.introspection.table_names()
    return tables 
