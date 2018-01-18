from django import forms

class CutterForm(forms.Form):
    center_address = forms.CharField(max_length=50)
    email = forms.EmailField(max_length=254)
    output_filename = forms.CharField(max_length=50)
    turf_count = forms.IntegerField()
    turf_size = forms.IntegerField()

    def clean(self):
        cleaned_data = super(CutterForm, self).clean()
        center_address = cleaned_data.get('center_address')
        email = cleaned_data.get('email')
        output_filename = cleaned_data.get('output_filename')
        turf_count = cleaned_data.get('tuft_count')
        turf_size = cleaned_data.get('turf_size')
        if not name and not email and not message:
            raise forms.ValidationError('You have to write something!')

