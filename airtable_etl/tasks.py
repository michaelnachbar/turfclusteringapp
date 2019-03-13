import string

from django.contrib.auth.models import User
from django.utils.crypto import get_random_string
from django.conf import settings

from celery import shared_task

from utilities import get_event_date, make_csv_to_form_dict, make_form_to_airtable_dict, update_columns, rename_columns

from airtable import Airtable
import pandas as pd
import datetime
import ciso8601


@shared_task
def upload_attendence_file(form,id):
    event_date = get_event_date(id)
    
    airtable_connection = Airtable('app1TMHqTJRUWnrvB', 'Event Attendance', api_key=settings.EVENT_ATTENDANCE_API_KEY)

    attendance_file = pd.read_csv('temp_attendence.csv')

    csv_to_form_dict = make_csv_to_form_dict(form)
    form_to_airtable_dict = make_form_to_airtable_dict()

    attendance_file = rename_columns(attendance_file,csv_to_form_dict,form_to_airtable_dict)
    attendance_file = update_columns(attendance_file,id)
    attendance_file = attendance_file.fillna("")

    attendance_file["Meeting Date"] = event_date
    attendance_file["Event Type"] = form["event_type"]

    airtable_connection.batch_insert(attendance_file.to_dict('records'))
