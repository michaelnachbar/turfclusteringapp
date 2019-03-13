from django import forms

from cutter.models import region

import pandas as pd


class AttendenceForm(forms.Form):
    attendence_file = forms.FileField(required=True)

def get_headers():
    data = pd.read_csv('temp_attendence.csv')
    headers = data.columns
    return [('N/A','N/A')] + [(i,i) for i in headers]

class FieldForm(forms.Form):
    #def __init__(self,*args,**kwargs):
        #google_id = kwargs.pop
    event_type = forms.CharField(help_text = "The event type (e.g. Meeting_GBM, Meeting_FAC, Canvas_M4A, Event_SNS)")
    email = forms.ChoiceField(choices = get_headers())
    first_name = forms.ChoiceField(choices = get_headers())
    last_name = forms.ChoiceField(choices = get_headers())
    phone_number = forms.ChoiceField(choices = get_headers())
    dsa_member = forms.ChoiceField(choices = get_headers())
    first_dsa_event = forms.ChoiceField(choices = get_headers())
    notes = forms.ChoiceField(choices = get_headers())
    



