from django.core.management.base import BaseCommand
import pandas as pd
from django.db import connection
from dermoral.models import Disease, Record, Medicine, Prescription  # Replace with your actual model

class Command(BaseCommand):
    help = 'Import data from Excel into Django model'

    def handle(self, *args, **options):
        Disease.objects.all().delete()
        Medicine.objects.all().delete()
        Prescription.objects.all().delete()
        Record.objects.all().delete()

        disease_path = 'C:/Users/ASUS/Desktop/Disease.xlsx'  # Replace with the actual file path
        treatment_path = 'C:/Users/ASUS/Desktop/Treatment.xlsx'  # Replace with the actual file path

        # Read Excel file into a DataFrame
        df = pd.read_excel(disease_path)
        tf = pd.read_excel(treatment_path)

        # Iterate through rows and create model instances
        for index, row in df.iterrows():
            dInstance = Disease(
                name=row['Name'],  # Replace with your column names
                description=row['Description'],
                symptom=row['Symptom'],
                cause=row['Cause'],
                # Add other fields as needed
            )
            dInstance.save()

        for index, row in tf.iterrows():
            tInstance = Medicine(
                name=row['Name'],  # Replace with your column names
                description=row['Description'],
                sideEffect=row['SideEffect'],
                # Add other fields as needed
            )
            tInstance.save()


        self.stdout.write(self.style.SUCCESS('Data imported successfully.'))
    
