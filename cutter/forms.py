from django import forms

from cutter.models import region

def get_regions():
    return [(j.name,j.name) for i,j in enumerate(region.objects.all())]

class CutterForm(forms.Form):
    center_address = forms.CharField(max_length=50,required=True)
    regions = get_regions()
    region_name = forms.ChoiceField(choices=regions)
    email = forms.CharField(max_length=50,required=True)
    turf_count = forms.IntegerField(required=True)
    turf_size = forms.IntegerField(required=True)

    """def clean(self):
        cleaned_data = super(CutterForm, self).clean()
        center_address = cleaned_data.get('center_address')
        email = cleaned_data.get('email')
        output_filename = cleaned_data.get('output_filename')
        turf_count = cleaned_data.get('tuft_count')
        turf_size = cleaned_data.get('turf_size')
        if not email:
            raise forms.ValidationError('You have to write something!')"""



class NewRegionForm(forms.Form):
    region_name = forms.CharField(max_length=50,required=True)
    email = forms.EmailField(max_length=254,required=True)
    open_addresses_io_file = forms.FileField(required=False)
    voter_file = forms.FileField(required=False)

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
