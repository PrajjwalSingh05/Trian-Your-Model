# Generated by Django 4.0.6 on 2023-04-06 01:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Train_model', '0002_trainedmodel_date_trainedmodel_time'),
    ]

    operations = [
        migrations.CreateModel(
            name='Feedback',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=40, null=True)),
                ('email', models.CharField(max_length=40, null=True)),
                ('feedback', models.CharField(max_length=40, null=True)),
            ],
        ),
    ]
