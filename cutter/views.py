from django.shortcuts import render
from .forms import CutterForm,NewRegionForm,UpdateRegionForm,BondCutterForm,AttendenceForm,FieldForm
from django.contrib import messages
from django.http import HttpResponseRedirect 
from django.core.urlresolvers import reverse



from tasks import output_turfs,add_region,region_update,bond_turfs

from cutter.models import region,region_progress

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

def bondcutter(request):
    if request.method == 'POST':
        form = BondCutterForm(request.POST,request.FILES)
        if 'skip_addresses_file' in request.POST and request.POST['skip_addresses_file']:
            with open('bond_skip_addresses.csv', 'wb+') as destination:
                for chunk in request.FILES['skip_addresses_file'].chunks():
                    destination.write(chunk)
        #if form.is_valid():
        bond_turfs.delay(form.data)
        messages.success(request, 'Thank you for submitting. You will get your turfs by email in 30 minutes or so.')
        print 'hey this is running'
        #else:
        #    print 'this is lame'
        #print form
    else:
        form = BondCutterForm()
    return render(request,'bondform.html', {'form': form})


def new_region(request):
    if request.method == 'POST':
        form = NewRegionForm(request.POST,request.FILES)
        #print form
        print form.__dict__
        print request.POST
        #if form.is_valid():
        
        if 'upload_new_files' in request.POST and request.POST['upload_new_files']:
            with open('temp_geocode_file_{region}.csv'.format(region=request.POST['region_name']), 'wb+') as destination:
                for chunk in request.FILES['open_addresses_io_file'].chunks():
                    destination.write(chunk)
            with open('temp_voter_file_{region}.csv'.format(region=request.POST['region_name']), 'wb+') as destination:
                for chunk in request.FILES['voter_file'].chunks():
                    destination.write(chunk)
        add_region.delay(form.data)
        messages.success(request, 'Thank you for submitting. Look for an email in a few minutes with next steps.')
        print 'hey this is running'
        #else:
            #print 'this is lame'
        #print form
    else:
        form = NewRegionForm()
    return render(request,'upload_form.html', {'form': form})

def update_region(request):
    if request.method == 'POST':
        print request.POST
        form = UpdateRegionForm(request.POST,request.FILES)
        region_update.delay(form.data)
        with open('temp_update_file_{region}.csv'.format(region=request.POST['region_name']), 'wb+') as destination:
            for chunk in request.FILES['update_file'].chunks():
                destination.write(chunk)
        messages.success(request, 'Thank you for submitting. Look for an email in a few minutes with next steps.')
        print 'hey this is running'
    else:
        form = UpdateRegionForm()
    return render(request,'update_form.html', {'form': form})


def attendence2(request,id="q"):
    if request.method == 'POST':
        print id
        print 'post___________________________'
        form = FieldForm(request.POST)
        if form.is_valid():
            upload_attendence_file.delay(form.data,id)
            messages.success(request, 'Thank you for submitting. Look for an email in a few minutes with next steps.')
    else:
        print id
        form = FieldForm()
        #print form
        print '_____________________________'
    return render(request,'attendence2.html', {'form': form})


def attendence(request,id="q"):
    if request.method == 'POST':
        print id
        print request.POST
        #form = AttendenceForm(request.POST)        
        form = AttendenceForm(request.POST,request.FILES)
        if form.is_valid():
            #region_update.delay(form.data)
            with open('temp_attendence.csv', 'wb+') as destination:
                for chunk in request.FILES['attendence_file'].chunks():
                    destination.write(chunk)
            return HttpResponseRedirect('/attendence2/' + id + "/")

        messages.success(request, 'Thank you for submitting. Look for an email in a few minutes with next steps.')
    else:
        form = AttendenceForm()
        print id
        print "__________"
    return render(request,'attendence.html', {'form': form,'google_id': id})



