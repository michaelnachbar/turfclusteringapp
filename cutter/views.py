from django.shortcuts import render
from .forms import CutterForm
from django.contrib import messages


from tasks import output_turfs

def cutter(request):
    if request.method == 'POST':
        form = CutterForm(request.POST)
        if form.is_valid():
            output_turfs.delay(form.cleaned_data)
            messages.success(request, 'Thank you for submitting. You will get your turfs by email in 30 minutes or so.')
            print 'hey this is running'
        else:
            print 'this is lame'
        #print form
    else:
        form = CutterForm()
    return render(request,'form.html', {'form': form})
