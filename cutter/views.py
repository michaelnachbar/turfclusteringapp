from django.shortcuts import render
from .forms import CutterForm

from tasks import output_turfs

def cutter(request):
    if request.method == 'POST':
        form = CutterForm(request.POST)
        if form.is_valid():
            output_turfs.delay(form.cleaned_data)
            print 'hey this is running'
        else:
            print 'this is lame'
        #print form
    else:
        form = CutterForm()
    return render(request,'form.html', {'form': form})
