from airtable import Airtable
import pandas as pd
import datetime
import ciso8601

from django.conf import settings

def get_event_date(event_code):
    airtable_client = Airtable('app1TMHqTJRUWnrvB', 'Events', api_key=settings.EVENTS_API_KEY)
    try:
        meeting_date = airtable_client.match('Event Code', str(event_code))['fields']['Date']
    except:
        meeting_date = ""
    return meeting_date

def make_csv_to_form_dict(form):
    ret = {}
    for key,val in form.iteritems():
        if val == "N/A":
            continue
        if key == u'csrfmiddlewaretoken':
            continue
        if key == "event_type":
            continue
        ret[key] = val
    return ret

def make_form_to_airtable_dict():
    ret = {
        "email": "Email",
        "meeting_date": "Meeting Date",
        "first_name": "First Name",
        "last_name": "Last Name",
        "phone_number": "Phone Number",
        "dsa_member": "DSA Member?",
        "first_dsa_event": "First DSA event?",
        "notes": "Notes"}
    return ret

def rename_columns(attendance_file,csv_to_form_dict,form_to_airtable_dict):
    for i in csv_to_form_dict.keys():
        attendance_file = attendance_file.rename(index=str, columns={csv_to_form_dict[i]: i})

    for i in attendance_file.columns:
        if not i in csv_to_form_dict.keys():
            attendance_file = attendance_file.drop(columns=i)

    attendance_file = attendance_file.rename(index=str,columns = form_to_airtable_dict)

    return attendance_file

def parse_tf(s):
    ret = False
    s = str(s)
    if s.upper() == "TRUE":
        ret = True
    if "Y" in s.upper():
        ret = True
    return ret


def update_dsa_member(row):
    row["DSA Member?"] = parse_tf(row["DSA Member?"])
    row["First DSA event?"] = parse_tf(row["First DSA event?"])
    return row

def update_event_link(row,*args,**kwargs):
    row["Event Link"] = [kwargs["event_code"]]
    return row


def update_columns(attendance_file,event_code):
    attendance_file = attendance_file.apply(update_dsa_member,axis=1)
    attendance_file = attendance_file.apply(update_event_link,axis=1,event_code=event_code)
    return attendance_file


