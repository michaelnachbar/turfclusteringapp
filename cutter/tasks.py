import string

from django.contrib.auth.models import User
from django.utils.crypto import get_random_string

from celery import shared_task

import pandas as pd
from fpdf import FPDF
import os
import shutil
from random import randint

from utilities import make_filtered_file, update_thresholds, load_threshold_dict, update_slice_data_nulls, update_slice_data_avgs,\
update_slice_data_clusters, update_slice_data_check_clusters, check_bad_streets, get_cluster_totals, update_cluster_numbers,\
split_cluster, make_html_file, get_street_list, text_page, make_img_file, add_img, new_whole_cluster, write_cluster, write_address_rows, \
write_assign_sheet, send_email 


@shared_task
def output_turfs(form):
    #Get the parameters from the Django form
    num_clusters = form['turf_count']
    turf_size = form['turf_size']
    center_address = form['center_address']
    filename = form['output_filename']
    email = form['email']
    
    folder_name = 'temp_folder_' + str(randint(1000,10000))


    os.makedirs(folder_name)
    os.makedirs(folder_name + '/temp_folder')
    
    #Updated data is the master list of addresses and # of registered voters
    #Will replace with an actual database in a future update
    data = pd.read_excel("Updated_data.xlsx")


    #Based on turf size and central point take the X closest addresses
    make_filtered_file(data,center_address,num_clusters,turf_size,folder_name + "/Test_filter.xlsx")


    #Take big data file out of memory
    data = None
    
    
    

    #Open filtered data    
    slice_data = pd.read_excel(folder_name + "/Test_filter.xlsx")
    
    #Open list of file with lists of 2 streets and if they intersect
    #Format of file is:
    #12th Street, River Street, TRUE
    #12th Street, 13th Street, FALSE
    intersect_data = pd.read_csv("Intersections_1.csv")



    #Look at the list of streets and find the intersections
    #This is used to ensure that we make continuous routes
    u = update_thresholds(slice_data)
    if not u:
        print 'Still need to collect more addresses'
        send_error_email(email)
        return


    #Load this list of intersections
    threshold_dict = load_threshold_dict(True)



    #print slice_data
    #Give addresses with nulls for voters and doors 0 for voters and doors
    slice_data = update_slice_data_nulls(slice_data)
    #Add avg lat and lon for street to each address. This will be used for clustering
    slice_data_updated = update_slice_data_avgs(slice_data)
    #Create clusters - assign a turf # to each address 
    slice_data_updated = update_slice_data_clusters(slice_data_updated,num_clusters)
    #Look for clusters that are not a continuous route.
    #Remove streets that don't connect from the cluster
    slice_data_updated = update_slice_data_check_clusters(slice_data_updated,num_clusters,threshold_dict)
    #For streets that got removed, try to find a new cluster
    slice_data_updated = check_bad_streets(slice_data_updated,threshold_dict)



    #Scroll through turfs and split turfs with too many doors into 2
    #As long as there are 2 turfs
    check_for_splits = True
    while check_for_splits:
        check_for_splits = False
        #Reorder clusters by proximity to center point
        slice_data_updated = update_cluster_numbers(slice_data_updated)
        #Cluster totals is a frame where each row is a cluster and contains cluster-level stats
        cluster_totals = get_cluster_totals(slice_data_updated)

        
        #Take the largest clusters and split them into 2
        for i in cluster_totals.itertuples():
            #Right now we're setting the max size of a turf as 2.4 doors * the turf size
            #And then subtract (.3 * the turf size) doors for each km of walking distance
            #We've found these numbers are OK for a 2.5 hour canvas, erring to the side of being too big
            #Def want to make these configurable
            max_size = 2.4 * turf_size - (.3 * turf_size * i.walking_distance)
            #Only split a turf if:
            #1. It has more doors than the max
            #2. It has more than 1 address (this is too keep it 1 team per address and avoid confusion)
            #(We never split into more than 2 turfs, it was just too messy when I tried it with more).
            splits = min(int(i.doors)/int(max_size/2),int(i.addresses),2)
            if splits > 1:
                #Split turf into 2
                slice_data_updated = split_cluster(slice_data_updated,i.Cluster,splits)
                check_for_splits = True

    #Remove no voter clusters
    for i in cluster_totals.itertuples():
        if i.voters == 0:
            #print i
            slice_data_updated.loc[slice_data_updated["Cluster"]==i.Cluster,"Cluster"] = -1



    


    #Reorder clusters by distance
    slice_data_updated = update_cluster_numbers(slice_data_updated)
    #Get cluster-level statistic
    cluster_totals = get_cluster_totals(slice_data_updated)

    #Scroll through clusters and try to merge small turfs with another turf
    missing_clusters = []
    for i in cluster_totals.itertuples():
        #Min desired turf size is 1.8 * turf_size - (.3 * turf size * walking distance in km)
        min_size = 1.8 * turf_size - (.3 * turf_size * i.walking_distance)
        if i.doors < (min_size) and i.addresses < (min_size):
            #Function checks for a potential merger
            upd_cluster = new_whole_cluster(cluster_totals,slice_data_updated,i.Cluster,threshold_dict,turf_size,missing_clusters)
            if upd_cluster:
                #If there's a merger update the cluster column on the list of addresses
                slice_data_updated.loc[(slice_data_updated.Cluster == i.Cluster) ,"Cluster"] = upd_cluster
                #Updated the list of merged clusters so we don't try to merge with a cluster that already merged
                missing_clusters.append(i.Cluster)


    #re-order clusters by distance
    slice_data_updated = update_cluster_numbers(slice_data_updated)
    #Get cluster level stats
    cluster_totals = get_cluster_totals(slice_data_updated)


    #Remove clusters that are too small
    for i in cluster_totals.itertuples():
        min_size = 1.2 * turf_size - (.3 * turf_size * i.walking_distance)
        if i.doors < (min_size):
            slice_data_updated.loc[slice_data_updated["Cluster"]==i.Cluster,"Cluster"] = -1


    #re-order clusters by distance
    slice_data_updated = update_cluster_numbers(slice_data_updated)
    #Get cluster level stats
    cluster_totals = get_cluster_totals(slice_data_updated)
    
    #Write list of addresses to file
    slice_data_updated.to_excel(folder_name + "/Cluster_data.xlsx")
    #Write list of turfs to file
    cluster_totals.to_excel(folder_name + "/Cluster_totals.xlsx")
    
    #Take files out of memory    
    slice_data_updated = None
    cluster_totals = None    


    #Read cluster data
    data = pd.read_excel(folder_name + "/Cluster_data.xlsx")

    #Create a PDF file
    pdf = FPDF()

    #Figure out how many clusters we have
    max_cluster = max(data["Cluster"])

    #For each cluster make a page on the PDF file
    #Each page will have a map with dots for each address
    #And a list of streets to hit
    #These will be printed and given to canvassers
    for cluster in range(max_cluster + 1):
        #For each cluster get the list of addresses in that cluster
        zoom_plot_data = data.loc[data["Cluster"]==cluster,:]
        zoom_plot_data = zoom_plot_data.reset_index()
        #Get the number of registered doors and voters
        doors = sum(zoom_plot_data['doors'])
        voters = sum(zoom_plot_data['voters'])
        #Make an html map from the list of addresses
        make_html_file(zoom_plot_data,folder_name)
        #List all the streets the canvasser will hit 
        street_list = get_street_list(zoom_plot_data)
        #Put text on the PDF with info about the turf
        text_page(pdf,cluster,street_list,doors,voters)
        #Convert the html file into an image file
        make_img_file(cluster,folder_name)
        #Put the image file onto the PDF
        add_img(pdf,cluster,folder_name,w=195)


    #Save the PDF
    pdf.output(folder_name + '/temp_folder/Turfs.pdf', 'F')
    

    
    #Make a new PDF file
    #This file will be a list of addresses for the canvassers to visit
    pdf=FPDF(format='letter', unit='in',orientation='L')
    pdf.set_fill_color(215)

    #Scroll through list of clusters and write the list to PDF
    for i in range(max_cluster + 1):
        write_cluster(pdf,data,i)

    #Write sheets to the PDF
    pdf.output(folder_name + '/temp_folder/Sheets.pdf', 'F')
    

    #Read cluster totals file
    data = pd.read_excel(folder_name + "/Cluster_totals.xlsx")

    #Make another PDF file
    #This will be the master sheet used to assign canvas teams
    #This is set up for teams of 2 and 4 depeding on turf size
    #Need to make this configurable
    pdf=FPDF(format='letter', unit='in',orientation='P')
    pdf.set_fill_color(25)
    pdf.add_page()
    pdf.set_font('Times','B',14.0) 
    pdf.cell(7, 0.0, "Team Assignment Sheet", align='C')
    pdf.ln(0.5)

    #Scroll through clusters and write data for each cluster
    for i in data.itertuples():
        write_assign_sheet(pdf,i,turf_size)

    pdf.output(folder_name + '/temp_folder/Assign_sheet.pdf', 'F')

    #Email 3 pdfs to email address specified
    send_email(email,folder_name)

    #Delete the temp folder
    shutil.rmtree(folder_name)
