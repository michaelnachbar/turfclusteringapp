# -*- coding: utf-8 -*-
# Generated by Django 1.9.4 on 2018-02-20 22:21
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cutter', '0005_bad_data'),
    ]

    operations = [
        migrations.AddField(
            model_name='bad_data',
            name='region',
            field=models.CharField(default='', max_length=100),
            preserve_default=False,
        ),
    ]
