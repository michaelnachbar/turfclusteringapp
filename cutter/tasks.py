import string

from django.contrib.auth.models import User
from django.utils.crypto import get_random_string

from celery import shared_task

import pandas as pd
from fpdf import FPDF
import os
import shutil

from utilities import make_filtered_file, update_thresholds, load_threshold_dict, update_slice_data_nulls, update_slice_data_avgs,\
update_slice_data_clusters, update_slice_data_check_clusters, check_bad_streets, get_cluster_totals, update_cluster_numbers,\
split_cluster, make_file, get_street_list, text_page, make_img_file, add_img, new_whole_cluster, write_cluster, write_address_rows, \
write_assign_sheet, send_email

@shared_task
def output_turfs(form):
    num_clusters = form['turf_count']
    turf_size = form['turf_size']
    center_address = form['center_address']
    filename = form['output_filename']
    email = form['email']

    os.makedirs('temp_folder')
    
    """
    # In[13]:

    data = pd.read_excel("Updated_data.xlsx")


    # In[27]:

    make_filtered_file(data,center_address,num_clusters,turf_size,"Test_filter.xlsx")


    # In[28]:

    data = None
    
    
    
    # In[29]:

    #Open filtered data    
    slice_data = pd.read_excel("Test_filter.xlsx")
    #Open list of file that states whether combinations of streets intersect
    intersect_data = pd.read_csv("Intersections_1.csv")


    # In[30]:

    #Check if streets intersect
    #This will be used to ensure that canvassers are given continuous routes
    u = update_thresholds(slice_data)
    if not u:
        print 'Still need to collect more addresses'
        send_error_email(email)
        return


    # In[31]:
    #Load this list of intersections
    threshold_dict = load_threshold_dict(True)


    # In[32]:
    #print slice_data
    #Give addresses with nulls for voters and doors 0 for voters and doors
    slice_data = update_slice_data_nulls(slice_data)
    #Add avg lat and lon for street to each address. This will be used for clustering
    slice_data_updated = update_slice_data_avgs(slice_data)
    #Create clusters - assign a turf # to each address 
    slice_data_updated = update_slice_data_clusters(slice_data_updated,num_clusters)
    slice_data_updated.to_excel("Test1.xlsx")
    #Look for clusters that are not a continuous route.
    #Remove streets that don't connect from the cluster
    slice_data_updated = update_slice_data_check_clusters(slice_data_updated,num_clusters,threshold_dict)
    slice_data_updated.to_excel("Test2.xlsx")
    #For streets that got removed, try to find a new cluster
    slice_data_updated = check_bad_streets(slice_data_updated,threshold_dict)
    slice_data_updated.to_excel("Test3.xlsx")
    #print slice_data_updated


    # In[ ]:


    check_for_splits = True
    while check_for_splits:
        check_for_splits = False
        # In[34]:
        #Add cluster level statistics to each address
        slice_data_updated = update_cluster_numbers(slice_data_updated)
        cluster_totals = get_cluster_totals(slice_data_updated)
        print "Is this working?"
        print slice_data_updated.head()
        print cluster_totals.head()
        #slice_data_updated.to_excel("Test3.5.xlsx")
        #print slice_data_updated
        
        #Scroll through clusters and split larger clusters into 2
        for i in cluster_totals.itertuples():
            max_size = 2.4 * turf_size - (30 * i.walking_distance)
            splits = min(int(i.doors)/int(max_size/2),int(i.addresses),2)
            print 'splitting'
            print splits
            print i.walking_distance
            print i.doors
            if splits > 1:
                slice_data_updated = split_cluster(slice_data_updated,i.Cluster,splits)
                check_for_splits = True
    slice_data_updated.to_excel("Test4.xlsx")
    #Remove no voter clusters
    for i in cluster_totals.itertuples():
        if i.voters == 0:
            #print i
            slice_data_updated.loc[slice_data_updated["Cluster"]==i.Cluster,"Cluster"] = -1



    


    #Add cluster level averages to each address
    slice_data_updated = update_cluster_numbers(slice_data_updated)
    cluster_totals = get_cluster_totals(slice_data_updated)

    missing_clusters = []
    for i in cluster_totals.itertuples():
        print i
        min_size = 1.8 * turf_size - (30 * i.walking_distance)
        if i.doors < (min_size) and i.addresses < (min_size):
            upd_cluster = new_whole_cluster(cluster_totals,slice_data_updated,i.Cluster,threshold_dict,turf_size,missing_clusters)
            if upd_cluster:
                slice_data_updated.loc[(slice_data_updated.Cluster == i.Cluster) ,"Cluster"] = upd_cluster
                missing_clusters.append(i.Cluster)


    slice_data_updated.to_excel("Test5.xlsx")

    # In[35]:

    slice_data_updated = update_cluster_numbers(slice_data_updated)
    cluster_totals = get_cluster_totals(slice_data_updated)
    #print 5
    #print slice_data_updated



    # In[36]:

    #Remove clusters that are too small
    for i in cluster_totals.itertuples():
        min_size = 1.2 * turf_size - (30 * i.walking_distance)
        if i.doors < (min_size):
            #print i
            slice_data_updated.loc[slice_data_updated["Cluster"]==i.Cluster,"Cluster"] = -1
    slice_data_updated.to_excel("Test6.xlsx")


    # In[37]:

    slice_data_updated = update_cluster_numbers(slice_data_updated)
    cluster_totals = get_cluster_totals(slice_data_updated)
    
    #print 6
    #print slice_data_updated



    # In[38]:

    slice_data_updated.to_excel("Cluster_data_1.10.xlsx")


    # In[39]:

    cluster_totals.to_excel("Cluster_totals_1.10.xlsx")
    
    return
    
    

    # In[22]:

    data = pd.read_excel("Cluster_data_1.10.xlsx")


    # In[24]:

    pdf = FPDF()


    # In[25]:
    max_cluster = max(data["Cluster"])

    for cluster in range(max_cluster):
        zoom_plot_data = data.loc[data["Cluster"]==cluster,:]
        zoom_plot_data = zoom_plot_data.reset_index()
        doors = sum(zoom_plot_data['doors'])
        voters = sum(zoom_plot_data['voters'])
        make_file(zoom_plot_data)
        street_list = get_street_list(zoom_plot_data)
        text_page(pdf,cluster,street_list,doors,voters)
        make_img_file(cluster)
        add_img(pdf,cluster,w=195)


    # In[27]:

    pdf.output('Turfs_1.21.pdf', 'F')
    return
    """

    data = pd.read_excel("Cluster_data_1.10.xlsx")
    max_cluster = max(data["Cluster"])

    pdf=FPDF(format='letter', unit='in',orientation='L')
    pdf.set_fill_color(215)

    for i in range(max_cluster):
        write_cluster(pdf,data,i)

    pdf.output('temp_folder/Sheets_1.21.pdf', 'F')
    
    data = pd.read_excel("Cluster_totals_1.10.xlsx")
    print data.head()
    pdf=FPDF(format='letter', unit='in',orientation='P')
    pdf.set_fill_color(25)
    pdf.add_page()
    pdf.set_font('Times','B',14.0) 
    pdf.cell(7, 0.0, "Team Assignment Sheet", align='C')
    pdf.ln(0.5)

    for i in data.itertuples():
        write_assign_sheet(pdf,i,turf_size)

    pdf.output('temp_folder/Assign_sheet_1.21.pdf', 'F')

    send_email(email)

    shutil.rmtree('temp_folder')
