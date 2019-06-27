from django import forms

from cutter.models import region

import pandas as pd

def get_regions():
    return [(j.name,j.name) for i,j in enumerate(region.objects.all())]

class CutterForm(forms.Form):
    center_address = forms.CharField(max_length=50,required=True)
    regions = get_regions()
    region_name = forms.ChoiceField(choices=regions)
    email = forms.CharField(max_length=50,required=True)
    turf_count = forms.IntegerField(required=True)
    turf_size = forms.IntegerField(required=True)
    extra_filters = forms.CharField(max_length=500,required=False)
    include_nonvoters = forms.BooleanField(help_text="Only applicable with extra filters",required=False)

    """def clean(self):
        cleaned_data = super(CutterForm, self).clean()
        center_address = cleaned_data.get('center_address')
        email = cleaned_data.get('email')
        output_filename = cleaned_data.get('output_filename')
        turf_count = cleaned_data.get('tuft_count')
        turf_size = cleaned_data.get('turf_size')
        if not email:
            raise forms.ValidationError('You have to write something!')"""

class BondCutterForm(forms.Form):
    center_address = forms.CharField(max_length=50,required=True)
    email = forms.CharField(max_length=50,required=True)
    est_canvassers = forms.IntegerField(required=True)
    percent_affordable = forms.IntegerField(required=True,help_text = "(Out of 100. If you are unsure put 70.)")
    skip_addresses_file = forms.FileField(required=False)


class AptCutterForm(forms.Form):
    center_address = forms.CharField(max_length=50,required=True)
    email = forms.CharField(max_length=50,required=True)
    est_canvas_teams = forms.IntegerField(required=True)
    skip_addresses_file = forms.FileField(required=False)


class NewRegionForm(forms.Form):
    region_name = forms.CharField(max_length=50,required=True)
    email = forms.EmailField(max_length=254,required=True)
    open_addresses_io_file = forms.FileField(required=False)
    voter_file = forms.FileField(required=False)
    generate_recs = forms.BooleanField(required=False,initial=True)
    upload_new_files = forms.BooleanField(required=False,initial=True)

def get_regions():
    return [(j.name,j.name) for i,j in enumerate(region.objects.all())]

class UpdateRegionForm(forms.Form):
    regions = get_regions()
    region_name = forms.ChoiceField(choices=regions)
    email = forms.EmailField(max_length=254,required=True)
    update_file = forms.FileField(required=False)

    def __init__(self, *args, **kwargs):
        super(UpdateRegionForm, self).__init__(*args, **kwargs)
        self.fields['region_name'] = forms.ChoiceField(
            choices=get_regions() )

class AttendenceForm(forms.Form):
    attendence_file = forms.FileField(required=True)

def get_headers():
    data = pd.read_csv('temp_attendence.csv')
    headers = data.columns
    return ['N/A','N/A'] + [[i,i] for i in headers]

class FieldForm(forms.Form):
    #def __init__(self,*args,**kwargs):
        #google_id = kwargs.pop
    email = forms.ChoiceField(choices = get_headers())
    meeting_date = forms.ChoiceField(choices = get_headers())
    event_type = forms.ChoiceField(choices = get_headers())
    first_name = forms.ChoiceField(choices = get_headers())
    last_name = forms.ChoiceField(choices = get_headers())
    phone_number = forms.ChoiceField(choices = get_headers())
    dsa_member = forms.ChoiceField(choices = get_headers())
    first_dsa_event = forms.ChoiceField(choices = get_headers())
    notes = forms.ChoiceField(choices = get_headers())
    




