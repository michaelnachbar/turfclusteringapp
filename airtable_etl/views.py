from django.shortcuts import render
from .forms import AttendenceForm,FieldForm
from django.contrib import messages
from django.http import HttpResponseRedirect 
from django.core.urlresolvers import reverse

import pandas as pd



from tasks import upload_attendence_file


def attendence2(request,id="q"):
    if request.method == 'POST':
        data = pd.read_csv('temp_attendence.csv')
        headers = data.columns
        headers = [('N/A','N/A')] + [(i,i) for i in headers]
        form = FieldForm(request.POST,headers=headers)
        if form.is_valid():
            upload_attendence_file.delay(form.data,id)
            messages.success(request, 'Thank you for submitting. Look for an email in a few minutes with next steps.')
    else:
        data = pd.read_csv('temp_attendence.csv')
        headers = data.columns
        headers = [('N/A','N/A')] + [(i,i) for i in headers]
        form = FieldForm(headers=headers)
    return render(request,'attendence2.html', {'form': form})


def attendence(request,id="q"):
    if request.method == 'POST':
        form = AttendenceForm(request.POST,request.FILES)
        if form.is_valid():
            with open('temp_attendence.csv', 'wb+') as destination:
                for chunk in request.FILES['attendence_file'].chunks():
                    destination.write(chunk)
            return HttpResponseRedirect('/attendence2/' + id + "/")

        messages.success(request, 'Thank you for submitting. Look for an email in a few minutes with next steps.')
    else:
        form = AttendenceForm()
    return render(request,'attendence.html', {'form': form,'google_id': id})
