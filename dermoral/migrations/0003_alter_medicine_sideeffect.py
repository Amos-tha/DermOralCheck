# Generated by Django 4.2.7 on 2023-11-25 16:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dermoral', '0002_alter_medicine_sideeffect'),
    ]

    operations = [
        migrations.AlterField(
            model_name='medicine',
            name='sideEffect',
            field=models.TextField(max_length=1000),
        ),
    ]
