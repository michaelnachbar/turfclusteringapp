# -*- coding: utf-8 -*-
# Generated by Django 1.9.4 on 2018-02-15 16:25
from __future__ import unicode_literals

from django.db import migrations, models
import django_mysql.models


class Migration(migrations.Migration):

    dependencies = [
        ('cutter', '0002_auto_20180215_0128'),
    ]

    operations = [
        migrations.CreateModel(
            name='voter_json',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('region', models.CharField(max_length=100)),
                ('address', models.CharField(max_length=100)),
                ('json_data', django_mysql.models.JSONField(default=dict)),
            ],
        ),
        migrations.AlterField(
            model_name='raw_geocode_address',
            name='region',
            field=models.CharField(max_length=100),
        ),
    ]
