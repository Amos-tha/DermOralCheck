from django.contrib import admin
from .models import Account, Disease, Medicine, Record, Prescription

# Register your models here.
admin.site.register(Account)
admin.site.register(Disease)
admin.site.register(Medicine)
admin.site.register(Record)
admin.site.register(Prescription)