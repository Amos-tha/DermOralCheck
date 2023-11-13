from django.core.management.base import BaseCommand
import pandas as pd
from django.db import connection
from dermoral.models import Disease, Record  # Replace with your actual model

class Command(BaseCommand):
    help = 'Import data from Excel into Django model'

    def handle(self, *args, **options):
        Disease.objects.all().delete()
        Record.objects.all().delete()

        file_path = 'C:/Users/ASUS/Desktop/SkinDisease.xlsx'  # Replace with the actual file path

        # Read Excel file into a DataFrame
        df = pd.read_excel(file_path)

        # Iterate through rows and create model instances
        for index, row in df.iterrows():
            instance = Disease(
                name=row['Name'],  # Replace with your column names
                description=row['Description'],
                symptom=row['Symptom'],
                cause=row['Cause'],
                # Add other fields as needed
            )
            instance.save()

        self.stdout.write(self.style.SUCCESS('Data imported successfully.'))
    
