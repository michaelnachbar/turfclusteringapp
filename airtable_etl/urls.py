from django.conf.urls import url
from django.contrib import admin

from cutter.views import cutter

urlpatterns = [
    url(r'^attendence/(.*?)/', 'airtable_etl.views.attendence'), 
    url(r'^attendence/(.*?)', 'airtable_etl.views.attendence'), 
    url(r'^attendence/', 'airtable_etl.views.attendence'),
    url(r'^attendence2/(.*?)/', 'airtable_etl.views.attendence2'),
    url(r'^attendence2/(.*?)', 'airtable_etl.views.attendence2'),    
    url(r'^attendence2/', 'airtable_etl.views.attendence2'),
    
]
