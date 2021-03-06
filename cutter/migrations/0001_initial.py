# -*- coding: utf-8 -*-
# Generated by Django 1.9.4 on 2018-02-14 22:06
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='raw_geocode_address',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('address', models.CharField(max_length=100)),
                ('number', models.IntegerField()),
                ('street', models.CharField(max_length=100)),
                ('LAT', models.FloatField()),
                ('LON', models.FloatField()),
            ],
        ),
    ]
