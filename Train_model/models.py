from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Signup(models.Model):
    user=models.ForeignKey(User,on_delete=models.CASCADE)
    # moblie=models.CharField(max_length=20,null=True)
    # # image=models.FileField(null=True)
    # gender=models.CharField(max_length=20,null=True)
    # dob=models.CharField(max_length=20,null=True)
    def  _str_(self):
        return self.user.username
class TrainedModel(models.Model):
    uniqueid=models.IntegerField(primary_key=True)
    username=models.CharField(max_length=20)
    location=models.CharField(max_length=200)
    modelname=models.CharField(max_length=40)
    date=models.CharField(max_length=30,null=True)
    time=models.TimeField(max_length=200)
# Create your models here.
class FlipkartMobileModel(models.Model):
    brand = models.CharField(max_length=100)
    price = models.CharField(max_length=100)
    
    memory = models.CharField(max_length=100)
    display = models.CharField(max_length=100)
    camera = models.CharField(max_length=100)
    battery = models.CharField(max_length=100)
    rom = models.CharField(max_length=100)
   

    def __str__(self):
        return self.brand
    

class FlipkartLaptopModel(models.Model):
    name= models.CharField(max_length=100)
    price = models.CharField(max_length=100)
    
    processor = models.CharField(max_length=100)
    ram = models.CharField(max_length=100)
    opearting_system = models.CharField(max_length=100)
    hard_disk= models.CharField(max_length=100)
    display= models.CharField(max_length=100)
   

    def __str__(self):
        return self.name

class FlipkartTelivisionModel(models.Model):
    name= models.CharField(max_length=100)
    price = models.CharField(max_length=100)
    display = models.CharField(max_length=100)
    rating = models.CharField(max_length=100)
    operating_system = models.CharField(max_length=100)
    warrently = models.CharField(max_length=100)

    def __str__(self):
        return self.name
    
class FlipkartEarphoneModel(models.Model):
    name= models.CharField(max_length=100)
    price = models.CharField(max_length=100)
    color = models.CharField(max_length=100)
    headphone_type = models.CharField(max_length=100)
    inline_remote = models.CharField(max_length=100)
    connectivity = models.CharField(max_length=100)
    # warrently = models.CharField(max_length=100)

    def __str__(self):
        return self.name
    

class FlipkartBikeModel(models.Model):
    name= models.CharField(max_length=100)
    price = models.CharField(max_length=100)
    displacement= models.CharField(max_length=100)
    max_power = models.CharField(max_length=100)
    mileage = models.CharField(max_length=100)
 

    def __str__(self):
        return self.name
    

class FlipkartWachingMachineModel(models.Model):
    brand = models.CharField(max_length=100)
    model= models.CharField(max_length=100)
    price = models.CharField(max_length=100)
    energy_rating = models.CharField(max_length=100)
    maximum_spin_speed = models.CharField(max_length=100)
    washing_capacity = models.CharField(max_length=100)
    # warrently = models.CharField(max_length=100)

    def __str__(self):
        return self.brand


