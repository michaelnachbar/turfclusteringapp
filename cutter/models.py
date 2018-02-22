from __future__ import unicode_literals

from django.db import models
from django_mysql.models import JSONField


# Create your models here.
class region(models.Model):
    name = models.CharField(max_length=100)


class voter_json(models.Model):
    region = models.CharField(max_length=100)
    address = models.CharField(max_length=100)
    json_data = JSONField()

class raw_geocode_address(models.Model):
    region = models.CharField(max_length=100)
    address = models.CharField(max_length=100)
    number = models.IntegerField()
    street = models.CharField(max_length=100)
    LAT = models.FloatField()
    LON = models.FloatField()

class canvas_data(models.Model):
    region = models.CharField(max_length=100)
    address = models.CharField(max_length=100)
    full_street = models.CharField(max_length=100)
    orig_address = models.CharField(max_length=100) 
    voters = models.IntegerField()
    doors = models.IntegerField()
    NUMBER = models.IntegerField()
    STREET = models.CharField(max_length=100)
    LAT = models.FloatField()
    LON = models.FloatField()

class bad_geo_data_failsafe(models.Model):
    region = models.CharField(max_length=100)
    address = models.CharField(max_length=100)
    full_street = models.CharField(max_length=100)
    orig_address = models.CharField(max_length=100) 
    voters = models.IntegerField()
    doors = models.IntegerField()
    NUMBER = models.IntegerField()
    STREET = models.CharField(max_length=100)
    LAT = models.FloatField()
    LON = models.FloatField()


class bad_data(models.Model):
    region = models.CharField(max_length=100)
    address = models.CharField(max_length=100)
    full_street = models.CharField(max_length=100)
    voters = models.IntegerField()
    doors = models.IntegerField()
    orig_address = models.CharField(max_length=100) 


class intersections(models.Model):
    street1 = models.CharField(max_length=100)
    street2 = models.CharField(max_length=100)
    distance = models.FloatField()
    lat = models.FloatField()
    lon = models.FloatField()
    region = models.CharField(max_length=100)
