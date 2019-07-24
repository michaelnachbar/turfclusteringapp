from __future__ import unicode_literals

from django.db import models
from django_mysql.models import JSONField


def install_sqlite_json_field():
    "Install sqlite hooks to load/dump json."
    import sqlite3
    import json

    def adapt_json(data):
        return (json.dumps(data, sort_keys=True)).encode()

    def convert_json(blob):
        return json.loads(blob.decode())

    sqlite3.register_adapter(dict, adapt_json)
    sqlite3.register_adapter(list, adapt_json)
    sqlite3.register_adapter(tuple, adapt_json)
    sqlite3.register_converter(str('JSON'), convert_json)
install_sqlite_json_field()

# JSONField is generic enough that we can use it for sqlite connections as well
# as long as the adapter and converter hooks are used.  So by-pass the mysql
# version check if the default database connection is sqlite.
JSONField_orig__check_mysql_version = JSONField._check_mysql_version
def _check_sqlite(self):
    from django.db import DEFAULT_DB_ALIAS, connections
    if getattr(connections[DEFAULT_DB_ALIAS], 'vendor', None) == 'sqlite':
        return []
    return JSONField_orig__check_mysql_version(self)
JSONField._check_mysql_version = _check_sqlite


# Create your models here.
class region(models.Model):
    name = models.CharField(max_length=100)

class region_progress(models.Model):
    name = models.CharField(max_length=100)
    voter_json_complete = models.BooleanField(default=False)
    bad_data_complete = models.BooleanField(default=False)
    canvas_data_complete = models.BooleanField(default=False)
    bad_geo_failsafe_data = models.BooleanField(default=False)
    last_street = models.CharField(max_length=100)


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
