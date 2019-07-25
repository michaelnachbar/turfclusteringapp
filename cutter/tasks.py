import os
import time
import string
import shutil
import itertools
from random import randint
from multiprocessing.dummy import Pool as ThreadPool 

import sqlalchemy
import censusgeocode

import pandas as pd
import numpy as np
from fpdf import FPDF

from celery import shared_task

from utilities import make_filtered_file, load_threshold_dict, update_thresholds, update_slice_data_nulls, update_slice_data_avgs,\
update_slice_data_clusters, update_slice_data_check_clusters, check_bad_streets, get_cluster_totals, update_cluster_numbers,\
split_cluster, make_html_file, get_street_list, text_page, make_img_file, add_img, new_whole_cluster, write_cluster, write_address_rows, \
write_assign_sheet, send_email, send_error_email, write_json_data, iterate_merge, get_street_change_recs, clean_dataframe, \
send_file_email, add_new_region, get_coverage_ratio, replace_list, \
send_error_report,send_nofile_email, matchmaker, mile_distance, add_score, get_coordinates, read_afford_units, split_afford_units, \
main_backup_afford_units, get_bonus_afford_units, get_market_rate_units, clean_market_rate_units, split_market_rate_units, \
main_backup_market_rate_units, get_bonus_market_rate_units, merge_units, match_frames, bonus_match, make_pdf, bond_assign_pdf, \
get_apartment_list, populate_missing_lat_lon, iterate_apts, random_function, get_order, add_pages, upload_apartment_list

from utilities.database import write_sql_data, read_sql_data, execute_sql, simple_query

from cutter.models import region_progress


@shared_task
def output_turfs(form):
    #Get the parameters from the Django form
    num_clusters = form['turf_count']
    turf_size = form['turf_size']
    
    region = form['region_name']
    center_address = form['center_address'] + " " + region
    email = form['email']
    if form['extra_filters']:
        extra_filters = form['extra_filters']
    else:
        extra_filters = None
    include_nonvoters = form['include_nonvoters']

    try:
        #send_error_email(email)
        


        folder_name = 'temp_folder_' + str(randint(1000,10000))


        os.makedirs(folder_name)
        os.makedirs(folder_name + '/temp_folder')
        
        #Updated data is the master list of addresses and # of registered voters
        #Will replace with an actual database in a future update
        #data = pd.read_excel("Updated_data.xlsx")
        #data = pd.read_excel("District_7_data.xlsx")
        data = read_sql_data("SELECT distinct region, address, full_street, orig_address, voters, doors, NUMBER, STREET, LAT, LON FROM cutter_canvas_data where region = '{region}'".format(region=region))
        print len(data)
        if extra_filters:
            query = """
            SELECT
                distinct address as vd_address
            FROM
                `voter_data_{region}` vd 
            WHERE
                {extra_filters}
            """.format(region = region,extra_filters = extra_filters)
            good_addresses = read_sql_data(query)
            print len(good_addresses)
            if not include_nonvoters:
                columns = data.columns
                data = data.merge(good_addresses,how="inner",left_on="orig_address",right_on="vd_address")
                data = data.loc[:,columns]
                print len(data)
            else:
                columns = data.columns
                v_data = data.loc[data["voters"]<>0,:]
                nv_data = data.loc[data["voters"]==0,:]
                v_data = v_data.merge(good_addresses,how="inner",left_on="orig_address",right_on="vd_address")
                v_data = v_data.loc[:,columns]
                data = v_data.append(nv_data)
                v_data,nv_data = None,None
        print data.columns
        print data.head()
        #data = read_sql_data("SELECT distinct region, address, full_street, orig_address, voters, doors, NUMBER, STREET, LAT, LON FROM cutter_canvas_data where region = 'Austin,  TX'")


        #Based on turf size and central point take the X closest addresses
        make_filtered_file(data,center_address,num_clusters,turf_size,folder_name + "/Test_filter.xlsx")


        #Take big data file out of memory
        data = None
        print 'made filtered file'
        
        
        

        #Open filtered data    
        slice_data = pd.read_excel(folder_name + "/Test_filter.xlsx")
        
        #Open list of file with lists of 2 streets and if they intersect
        #Format of file is:
        #12th Street, River Street, TRUE
        #12th Street, 13th Street, FALSE
        #intersect_data = pd.read_csv("Intersections_1.csv")



        #Look at the list of streets and find the intersections
        #This is used to ensure that we make continuous routes
        u = update_thresholds(slice_data,region)
        if not u:
            print 'Still need to collect more addresses'
            send_error_email(email)
            return


        #Load this list of intersections
        threshold_dict = load_threshold_dict(region,True)
        print 'loaded threshold dict'



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
        print 'clusters'

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
                max_size = 2 * turf_size - (.25 * turf_size * i.walking_distance)
                #Only split a turf if:
                #1. It has more doors than the max
                #2. It has more than 1 address (this is too keep it 1 team per address and avoid confusion)
                #(We never split into more than 2 turfs, it was just too messy when I tried it with more).
                splits = min(int(i.doors)/int(max_size/2),int(i.addresses),2)
                splits = max(splits,1)
                if splits > 1:
                    #Split turf into 2
                    slice_data_updated = split_cluster(slice_data_updated,i.Cluster,splits)
                    check_for_splits = True

        #Remove no voter clusters
        for i in cluster_totals.itertuples():
            if i.voters == 0:
                #print i
                slice_data_updated.loc[slice_data_updated["Cluster"]==i.Cluster,"Cluster"] = -1



        print 'splits'


        #Reorder clusters by distance
        slice_data_updated = update_cluster_numbers(slice_data_updated)
        #Get cluster-level statistic
        cluster_totals = get_cluster_totals(slice_data_updated)

        
        #Scroll through clusters and try to merge small turfs with another turf
        missing_clusters = []
        for i in cluster_totals.itertuples():
            if i.Cluster in missing_clusters:
                continue
            #Min desired turf size is 1.8 * turf_size - (.3 * turf size * walking distance in km)
            min_size = 1.4 * turf_size - (.25 * turf_size * i.walking_distance)
            if i.doors < (min_size) and i.addresses < (min_size):
                #Function checks for a potential merger
                upd_cluster = new_whole_cluster(cluster_totals,slice_data_updated,i.Cluster,threshold_dict,turf_size,missing_clusters)
                if upd_cluster:
                    #If there's a merger update the cluster column on the list of addresses
                    slice_data_updated.loc[(slice_data_updated.Cluster == i.Cluster) ,"Cluster"] = upd_cluster
                    #Updated the list of merged clusters so we don't try to merge with a cluster that already merged
                    missing_clusters.append(i.Cluster)
                    missing_clusters.append(upd_cluster)


        print 'merging'
        #Reorder clusters by distance
        slice_data_updated = update_cluster_numbers(slice_data_updated)
        #Get cluster-level statistic
        cluster_totals = get_cluster_totals(slice_data_updated)

        
        #Scroll through clusters and try to merge small turfs with another turf
        missing_clusters = []
        for i in cluster_totals.itertuples():
            if i.Cluster in missing_clusters:
                continue
            #Min desired turf size is 1.8 * turf_size - (.3 * turf size * walking distance in km)
            min_size = 1.4 * turf_size - (.25 * turf_size * i.walking_distance)
            if i.doors < (min_size) and i.addresses < (min_size):
                #Function checks for a potential merger
                upd_cluster = new_whole_cluster(cluster_totals,slice_data_updated,i.Cluster,threshold_dict,turf_size,missing_clusters)
                if upd_cluster:
                    #If there's a merger update the cluster column on the list of addresses
                    slice_data_updated.loc[(slice_data_updated.Cluster == i.Cluster) ,"Cluster"] = upd_cluster
                    #Updated the list of merged clusters so we don't try to merge with a cluster that already merged
                    missing_clusters.append(i.Cluster)
                    missing_clusters.append(upd_cluster)
        


        #re-order clusters by distance
        slice_data_updated = update_cluster_numbers(slice_data_updated)
        #Get cluster level stats
        cluster_totals = get_cluster_totals(slice_data_updated)

        print 'were close'
        #Remove clusters that are too small
        for i in cluster_totals.itertuples():
            min_size = .45 * turf_size
            if i.doors < (min_size):
                slice_data_updated.loc[slice_data_updated["Cluster"]==i.Cluster,"Cluster"] = -1


        #re-order clusters by distance
        slice_data_updated = update_cluster_numbers(slice_data_updated)
        #Get cluster level stats
        cluster_totals = get_cluster_totals(slice_data_updated)
        
        #Write list of addresses to file
        slice_data_updated.to_excel(folder_name + "/temp_folder/Cluster_data.xlsx")
        #Write list of turfs to file
        cluster_totals.to_excel(folder_name + "/Cluster_totals.xlsx")
        
        #Take files out of memory    
        slice_data_updated = None
        cluster_totals = None    


        #Read cluster data
        data = pd.read_excel(folder_name + "/temp_folder/Cluster_data.xlsx")
        print 'making pdf'
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
            add_img(pdf,cluster,folder_name,w=150)


        #Save the PDF
        pdf.output(folder_name + '/temp_folder/Turfs.pdf', 'F')
        print 'pdf saved'
        

        
        #Make a new PDF file
        #This file will be a list of addresses for the canvassers to visit
        pdf=FPDF(format='letter', unit='in',orientation='P')
        pdf.set_fill_color(215)
        pdf.set_auto_page_break(auto = True, margin = 0.1)

        #Scroll through list of clusters and write the list to PDF
        for i in range(max_cluster + 1):
            print i
            try:
                write_cluster(pdf,data,i)
            except:
                print data[data["Cluster"]==i]
        print 'pdf 2 saved'

        #Write sheets to the PDF
        pdf.output(folder_name + '/temp_folder/Sheets.pdf', 'F')
        print 'pdf 3 saved'
        

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

        print 'assigned sheets'

        pdf.output(folder_name + '/temp_folder/Assign_sheet.pdf', 'F')
        print 'saved file'

        #Email 3 pdfs to email address specified
        send_email(email,folder_name)
        print 'sent email'

        #Delete the temp folder
        shutil.rmtree(folder_name)
        print 'deleted file'
    except Exception as e:
        send_error_report(email,e)


@shared_task
def add_region(form):
    print form
    region = form['region_name']
    email = form['email']

    
    add_new_region(region)
    progress = region_progress.objects.get(name=region) 
    try:
        #Read list of registered voters 
        full_voter_data = pd.read_csv('temp_voter_file_{region}.csv'.format(region=region),chunksize=5000,dtype='str',keep_default_na=False)
        
        string_len_dict = {}
        varchar_len_dict = {}        
        for voter_data in full_voter_data:
            
            for col in voter_data.columns:
                #try:
                if not col in string_len_dict:
                    string_len_dict[col] = max(voter_data[col].str.len().max(),10)
                else:
                    string_len_dict[col] = max(string_len_dict[col],voter_data[col].str.len().max())  
        for col in string_len_dict.keys():
            varchar_len_dict[col] = sqlalchemy.types.NVARCHAR(length=string_len_dict[col])
        varchar_len_dict["full_street"] = sqlalchemy.types.NVARCHAR(length=1000)
        varchar_len_dict["address"] = sqlalchemy.types.NVARCHAR(length=1000)
        varchar_len_dict["address_exp"] = sqlalchemy.types.NVARCHAR(length=1000)


        #voter_data["state"] = "TX"
        start = True
        
        full_voter_data = pd.read_csv('temp_voter_file_{region}.csv'.format(region=region),chunksize=5000,dtype='str',keep_default_na=False)
        for voter_data in full_voter_data:
            voter_data.loc[:,("city","state","zip","BLKNUM","STRNAM","STRTYP","UNITYP","UNITNO")] = \
                voter_data.loc[:,("city","state","zip","BLKNUM","STRNAM","STRTYP","UNITYP","UNITNO")].fillna("")

     
            #Create address columns for voter data
            voter_data["STRNAM"] = voter_data["STRNAM"].str.upper()
            voter_data["STRNAM"] = voter_data["STRNAM"].str.strip()
            voter_data["STRTYP"] = voter_data["STRTYP"].str.upper()
            voter_data["STRTYP"] = voter_data["STRTYP"].str.strip()
            voter_data["BLKNUM"] = voter_data["BLKNUM"].map(str).str.replace("[^0-9]","")
            voter_data["address"] = voter_data["BLKNUM"] + \
                " " + voter_data["STRNAM"].map(str) + " " + voter_data["STRTYP"].map(str)
            voter_data["address"] = voter_data["address"].str.strip()
            voter_data["address_exp"] = voter_data["address"].map(str) + " " + voter_data["UNITYP"].map(str) + \
                " " + voter_data["UNITNO"].map(str)
            voter_data["full_street"] = voter_data["STRNAM"].map(str) + " " + voter_data["STRTYP"].map(str)
            voter_data["full_street"] = voter_data["full_street"].str.strip()
            voter_data["full_street"] = voter_data["full_street"].str.replace("STREET","ST")
            voter_data["full_street"] = voter_data["full_street"].str.replace("AVENUE","AVE")


            
                        

            if start:
                if not progress.voter_json_complete:
                    
                    try:
                        write_sql_data(voter_data,"voter_data_" + region,region,'replace',dtype=varchar_len_dict)
                    except Exception as e:
                        print type(e)     # the exception instance
                        print e.args      # arguments stored in .args
                        print e           # __str__ allows args to be printed directly
                start = False
                new_data = voter_data.loc[:,("city","state","zip","BLKNUM","address","address_exp","full_street")]
            else:
                if not progress.voter_json_complete:
                    try:
                        write_sql_data(voter_data,"voter_data_" + region,region,'append')
                    except Exception as e:
                        print type(e)     # the exception instance
                        print e.args      # arguments stored in .args
                        print e           # __str__ allows args to be printed directly
                new_data = new_data.append(voter_data.loc[:,("city","state","zip","BLKNUM","address","address_exp","full_street")])         
        progress.voter_json_complete = True
            

        

        #Cut voter data down to needed columns
        #new_data = voter_data.loc[:,("city","state","zip","BLKNUM","address","address_exp","full_street")]
        new_data = new_data.fillna("")
        new_data["zip"] = new_data["zip"].map(str).str[:5]

        voter_data = None

        #Read Geocoded List of Addresses
        print 'about to geocode'
        geocode_data = pd.read_csv('temp_geocode_file_{region}.csv'.format(region=region))
        geocode_data["STREET"] = geocode_data["STREET"].str.upper()
        geocode_data["STREET"] = geocode_data["STREET"].str.replace("STREET","ST")
        geocode_data["STREET"] = geocode_data["STREET"].str.replace("AVENUE","AVE")
        geocode_data["STREET"] = geocode_data["STREET"].str.replace("PLACE","PL")
        geocode_data["STREET"] = geocode_data["STREET"].str.replace("COURT","CT")
        geocode_data["NUMBER"] = geocode_data["NUMBER"].map(int).map(str)
        print 'geocode data'
        

        #Make a pivot of #of registered doors and registered voters per street address
        #new_data["complete_address"] = new_data["address"] + ", " + new_data["city"] + ", " + \
        #    new_data["state"] + ", " + new_data["zip"].map(str)
        f = {"address_exp": 'nunique', "address_exp": 'count'}
        new_addresses = new_data.groupby(["city","state","zip","BLKNUM","address","full_street"])["address_exp"].agg(['count','nunique'])
        new_addresses = new_addresses.reset_index()
        new_addresses.columns = ["city","state","zip","BLKNUM",'address',"full_street",'voters','doors']
        new_addresses["orig_address"] = new_addresses["address"]

        address_count = len(new_addresses.index)

        new_data=None

        filecount = 1+len(new_addresses) / 100
        new_addresses["complete_address"] = new_addresses["address"] + ", " + new_addresses["city"] + ", " + \
            new_addresses["state"] + ", " + new_addresses["zip"].map(str)

        def batch_function(num):
            try:
                time.sleep(num%100)
                temp_filename = "dummy{num}.csv".format(num=num)
                new_addresses.loc[num*100:num*100+99,("address","city","state","zip")].to_csv(temp_filename)
                z=censusgeocode.addressbatch(temp_filename)
                os.remove(temp_filename)
                print num
                return [[i['address'],i['lat'],i['lon']] for i in z]
            except:
                os.remove(temp_filename)
                print "error on {num}".format(num=num)

        array = range(filecount)
        pool = ThreadPool(100) 
        mapped_data = pool.map(batch_function,array)
        ret_list = list(itertools.chain.from_iterable(mapped_data))[1:]

        geo_voter_data = pd.DataFrame(ret_list,columns = ["complete_address","LAT","LON"])
        geo_voter_data = geo_voter_data.merge(right=new_addresses,how="inner",on="complete_address")
        geo_voter_data["LAT"] = geo_voter_data["LAT"].fillna(0)
        geo_voter_data["LON"] = geo_voter_data["LON"].fillna(0)

        #Create address column for geocoded data and cut down to neede columns
        geocode_data["LAT1"] = geocode_data.apply(lambda x: "{0:.2f}".format(x["LAT"]),axis=1)
        geocode_data["LON1"] = geocode_data.apply(lambda x: "{0:.2f}".format(x["LON"]),axis=1)
        geo_voter_data["LAT1"] = geo_voter_data.apply(lambda x: "{0:.2f}".format(x["LAT"]),axis=1)
        geo_voter_data["LON1"] = geo_voter_data.apply(lambda x: "{0:.2f}".format(x["LON"]),axis=1)
        geocode_data["LAT2"] = geocode_data.apply(lambda x: "{0:.3f}".format(x["LAT"]),axis=1)
        geocode_data["LON2"] = geocode_data.apply(lambda x: "{0:.3f}".format(x["LON"]),axis=1)
        geo_voter_data["LAT2"] = geo_voter_data.apply(lambda x: "{0:.3f}".format(x["LAT"]),axis=1)
        geo_voter_data["LON2"] = geo_voter_data.apply(lambda x: "{0:.3f}".format(x["LON"]),axis=1)

        geocode_data["first_letter"] = geocode_data["STREET"].str[0]
        geo_voter_data["first_letter"] = geo_voter_data["full_street"].str[0]
     
        voter_merge = geo_voter_data.loc[:,("BLKNUM","LAT1","LON1","first_letter")]
        geocode_missing = geocode_data.merge(voter_merge,how="left",left_on = ("NUMBER","LAT1","LON1","first_letter"), \
                                             right_on = ("BLKNUM","LAT1","LON1","first_letter"))
        geocode_missing = geocode_missing[pd.isnull(geocode_missing["BLKNUM"])]

        voter_merge = geo_voter_data.loc[:,("BLKNUM","LAT2","LON2")]
        geocode_missing = geocode_missing.merge(geo_voter_data,how="left",left_on = ("NUMBER","LAT2","LON2"), \
                                             right_on = ("BLKNUM","LAT2","LON2"))
        geocode_missing = geocode_missing[pd.isnull(geocode_missing["BLKNUM_y"])]

        geo_voter_data["region"] = region
        geo_voter_data = geo_voter_data.loc[:,("region","address", "full_street", "orig_address", "voters", \
                                               "doors", "BLKNUM", "full_street", "LAT", "LON")]

        geocode_missing["region"] = region
        geocode_missing["doors"] = 0
        geocode_missing["voters"] = 0
        geocode_missing["address"] = geocode_missing["NUMBER"].map(str) + " " + geocode_missing["STREET"]
        geocode_missing = geocode_missing.loc[:,("region", "address", "STREET", "address", "voters", \
                                                "doors", "NUMBER", "STREET", "LAT_x", "LON_x")]

        geo_voter_data.columns = ["region","address","full_street","orig_address","voters","doors","NUMBER","STREET","LAT","LON"]
        geocode_missing.columns = ["region","address","full_street","orig_address","voters","doors","NUMBER","STREET","LAT","LON"]

        new_address_count = len(geo_voter_data[pd.notnull(geo_voter_data["LAT"])].index)
        address_perc = 100 * new_address_count/address_count
        address_perc = "{0:.2f}".format(address_perc)
        geocode_missing_count = len(geocode_missing.index)

        email_text = """We were able to code {new_address_count} out of {address_count} addresses. Or 
            {perc}%. We added {geocode_missing} addresses on top of that. 
        """.format(new_address_count = str(new_address_count),address_count = str(address_count), \
            perc = address_perc, geocode_missing = str(geocode_missing_count))

        send_nofile_email(email,email_text)
        
        combo_data = geo_voter_data.append(geocode_missing)
        
        combo_data["NUMBER"]  = combo_data["NUMBER"].str.replace("[^0-9]","")
        combo_data["NUMBER"] = combo_data["NUMBER"].fillna(0)
        combo_data.loc[combo_data["NUMBER"]=="","NUMBER"] = 0
        combo_data["NUMBER"] = combo_data["NUMBER"].map(int)
        combo_data.loc[:,("region","address","full_street","orig_address","STREET")] = \
            combo_data.loc[:,("region","address","full_street","orig_address","STREET")].fillna("")
        write_sql_data(combo_data,'cutter_canvas_data',region)
        progress.canvas_data_complete = True
        
        
        progress.save()
    except Exception as e:
        send_error_report(email,e)
        progress.save()    


@shared_task
def add_region_2(form):
    print form
    region = form['region_name']
    email = form['email']
    if 'generate_recs' in form.keys():
        generate_recs = form['generate_recs']
    else:
        generate_recs = None

    
    add_new_region(region)
    progress = region_progress.objects.get(name=region) 
    try:
        #Read list of registered voters 
        voter_data = pd.read_csv('temp_voter_file_{region}.csv'.format(region=region))
        print len(voter_data)


        voter_data.loc[:,("BLKNUM","STRNAM","STRTYP","UNITYP","UNITNO")] = \
            voter_data.loc[:,("BLKNUM","STRNAM","STRTYP","UNITYP","UNITNO")].fillna("")

        #Create address columns for voter data
        voter_data["STRNAM"] = voter_data["STRNAM"].str.upper()
        voter_data["STRNAM"] = voter_data["STRNAM"].str.strip()
        voter_data["STRTYP"] = voter_data["STRTYP"].str.upper()
        voter_data["STRTYP"] = voter_data["STRTYP"].str.strip()
        voter_data["address"] = voter_data["BLKNUM"].map(str).str.replace("\.0","") + \
            " " + voter_data["STRNAM"].map(str) + " " + voter_data["STRTYP"].map(str)
        voter_data["address"] = voter_data["address"].str.strip()
        voter_data["address_exp"] = voter_data["address"].map(str) + " " + voter_data["UNITYP"].map(str) + \
            " " + voter_data["UNITNO"].map(str)
        voter_data["full_street"] = voter_data["STRNAM"].map(str) + " " + voter_data["STRTYP"].map(str)
        voter_data["full_street"] = voter_data["full_street"].str.strip()
        print len(voter_data)
        print voter_data.head()

        if not progress.voter_json_complete:
            #print "Writing JSON Data"
            #write_json_data(voter_data,\
            #    voter_data.columns.difference(['address','full_street','address_exp',"BLKNUM","STRNAM","STRTYP","STRDIR","UNITYP","UNITNO"]),\
            #    region)
            print "JSON Data Written"
            progress.voter_json_complete = True

        

        #Cut voter data down to needed columns
        new_data = voter_data.loc[:,("address","address_exp","full_street")]
        new_data = new_data.fillna("")

        voter_data = None

        #Read Geocoded List of Addresses
        geocode_data = pd.read_csv('temp_geocode_file_{region}.csv'.format(region=region))

        #Create address column for geocoded data and cut down to neede columns
        geocode_data["NUMBER"] = geocode_data["NUMBER"].map(str)
        geocode_data["NUMBER"] = geocode_data["NUMBER"].str.extract('(\d+)', expand=False)
        print 'filtered for #s'
        geocode_data["NUMBER"] = geocode_data["NUMBER"].fillna('0')
        print 'filled nas'
        geocode_data["NUMBER"] = geocode_data["NUMBER"].map(int)
        print 'converted to int'
        geocode_data["STREET"] = geocode_data["STREET"].str.upper()
        geocode_data["STREET"] = geocode_data["STREET"].str.strip()
        geocode_data["address"] = geocode_data["NUMBER"].map(str) + " " + geocode_data["STREET"]
        print 'made street names'
        geocode_data = geocode_data.loc[:,("address","LON","LAT","STREET","NUMBER")].groupby("address").max()
        geocode_data = geocode_data.reset_index()

        #Make a pivot of #of registered doors and registered voters per street address
        f = {"address_exp": 'nunique', "address_exp": 'count'}
        new_addresses = new_data.groupby(["address","full_street"])["address_exp"].agg(['count','nunique'])
        new_addresses = new_addresses.reset_index()
        new_addresses.columns = ['address',"full_street",'voters','doors']
        new_addresses["orig_address"] = new_addresses["address"]

        new_data=None

        results_dict = iterate_merge(geocode_data,new_addresses,None)
     
        if not progress.bad_data_complete:
            write_sql_data(results_dict['bad_data'],'cutter_bad_data',region)
            progress.bad_data_complete = True
        if not progress.canvas_data_complete:
            combo_data = results_dict['good_data'].append(results_dict['bad_full_geo_data'],ignore_index=True)
            print len(results_dict['good_data'])
            print len(results_dict['bad_full_geo_data'])
            print len(combo_data)
            combo_data.loc[:,("doors","voters","NUMBER")] = combo_data.loc[:,("doors","voters","NUMBER")].fillna(0)
            combo_data.loc[:,("full_street","orig_address")] = combo_data.loc[:,("full_street","orig_address")].fillna("")
            print combo_data.iloc[6268:6274,:]
            print max(combo_data["NUMBER"])
            print min(combo_data["NUMBER"])
            combo_data.to_csv("Test.csv")
            write_sql_data(combo_data,'cutter_canvas_data',region)
            progress.canvas_data_complete = True

        

        
        print "Getting Recs"
        if generate_recs:
            get_street_change_recs(results_dict['bad_data'],geocode_data=geocode_data,region=region)
        #street_change_recs = get_street_change_recs(results_dict['bad_data'],geocode_data=geocode_data,region=region)
        #pd.DataFrame(street_change_recs).to_excel("Change_recs.xlsx")
        
        x = str(get_coverage_ratio(region)) + " % of the voter data was geocoded."
        
        if generate_recs:
            send_file_email(email,"Change_recs_{region}.csv".format(region=region),x)
            os.remove("Change_recs_{region}.csv".format(region=region))
        else:
            send_file_email(email,None,x)
        
        progress.save()
    except Exception as e:
        send_error_report(email,e)
        progress.save()    
  

@shared_task
def region_update(form):
    
    email = form['email']  
    region = form['region_name']
    try:
        orig_ratio = get_coverage_ratio(region)
        bad_data = read_sql_data("SELECT * FROM cutter_bad_data where region = '{region}'".format(region=region))
        bad_geo_data = read_sql_data("""SELECT address,LON,LAT,STREET,NUMBER FROM cutter_canvas_data where full_street = ""
            and region = '{region}'""".format(region=region))

        print len(bad_data)
        print len(bad_geo_data)
        results_dict = iterate_merge(bad_geo_data,bad_data,replace_list,filename="top100.xlsx",replace_file = 'temp_update_file_{region}.csv'.format(region=region))

        print len(results_dict['good_data'])

        results_dict['bad_full_geo_data'].loc[:,("doors","voters")] = results_dict['bad_full_geo_data'].loc[:,("doors","voters")].fillna(0)
        results_dict['bad_full_geo_data'].loc[:,("full_street","orig_address")] = results_dict['bad_full_geo_data'].loc[:,("full_street","orig_address")].fillna("")
        write_sql_data(results_dict['bad_full_geo_data'],'cutter_bad_geo_data_failsafe',region)
        write_sql_data(results_dict['bad_data'],'cutter_bad_data',region,'replace')

        write_sql_data(results_dict['good_data'],'cutter_canvas_data',region)    
        
        
        
        execute_sql("DELETE FROM cutter_canvas_data WHERE full_street = '' and region = '{region}'".format(region=region))
        write_sql_data(results_dict['bad_full_geo_data'],'cutter_canvas_data',region)
        execute_sql("DELETE FROM cutter_bad_geo_data_failsafe WHERE full_street = '' and region = '{region}'".format(region=region))

        new_ratio = get_coverage_ratio(region)
        email_text = "Thank you for your corrections. They took us from {perc1}% of all registered voters geocoded to {perc2}% of all registered voters geocoded. Attached is a list of some of the streets missing coverage.".format(perc1=orig_ratio,perc2=new_ratio)
        send_file_email(email,"top100.xlsx",email_text)
        os.remove("top100.xlsx")
    except Exception as e:
        send_error_report(email,e)

@shared_task
def bond_turfs(form):

    folder_name = 'temp_folder_' + str(randint(1000,10000))


    os.makedirs(folder_name)
    os.makedirs(folder_name + '/temp_folder')

    center_address = form['center_address']
    email = form['email']
    est_canvassers = int(form['est_canvassers'])
    percent_affordable = float(form['percent_affordable']) / 100.0
    coords = get_coordinates(center_address,False)
    skip_addresses = pd.read_csv("bond_skip_addresses.csv")
    print form
    print coords
    
    afford_units = read_afford_units(coords,skip_addresses)
    big_afford_units,small_afford_units = split_afford_units(afford_units)

    main_afford_units,backup_afford_units = main_backup_afford_units(big_afford_units,est_canvassers,percent_affordable)
    bonus_afford_units = get_bonus_afford_units(small_afford_units,main_afford_units)
    

    market_rate_units = get_market_rate_units(coords,afford_units,skip_addresses)
    market_rate_units = clean_market_rate_units(market_rate_units,coords)
    big_market_rate_units,small_market_rate_units = split_market_rate_units(market_rate_units)
    

    main_market_rate_units,backup_market_rate_units = main_backup_market_rate_units(big_market_rate_units,percent_affordable,est_canvassers)
    bonus_market_rate_units = get_bonus_market_rate_units(small_market_rate_units,main_market_rate_units)
    

    main_units = merge_units(main_afford_units,main_market_rate_units)
    main_units = main_units.sort_values("score",ascending=False).reset_index(drop=True)
    backup_units = merge_units(backup_afford_units,backup_market_rate_units)
    bonus_units = merge_units(bonus_afford_units,bonus_market_rate_units)
    

    backup_dict = match_frames(main_units,backup_units,colname = "bigcount")
    main_unit_match,bonus_unit_match = bonus_match(main_units,bonus_units)
    bonus_dict = match_frames(main_unit_match,bonus_unit_match)

    make_pdf(main_units,backup_units,bonus_units,backup_dict,main_unit_match,bonus_unit_match,bonus_dict,folder_name)
    bond_assign_pdf(main_units,folder_name)

    #Email 3 pdfs to email address specified
    send_email(email,folder_name)
    print 'sent email'

    #Delete the temp folder
    shutil.rmtree(folder_name)

    return


@shared_task
def apt_turfs(form):
    folder_name = 'temp_folder_' + str(randint(1000,10000))


    os.makedirs(folder_name)
    os.makedirs(folder_name + '/temp_folder')

    center_address = form['center_address']
    email = form['email']
    est_canvas_teams = int(form['est_canvas_teams'])
    center_coords = get_coordinates(center_address,False)
    team_max = 45
    skip_addresses = pd.read_csv("bond_skip_addresses.csv")
    print 'Set coords'

    #Filter by region
    #Filter by skip addresses


    data = upload_apartment_list()
    print 'Got apartment list'

    

    data = data.loc[:,["address","units","cost","year","LAT","LON"]]

    #Fill in the missing year and cost with the averages
    avgyear = np.mean(data.loc[pd.notnull(data["year"]),"year"].map(float))
    avgcost = np.mean(data.loc[pd.notnull(data["cost"]),"cost"].map(float))
    #print avgyear
    data.loc[pd.isnull(data["cost"]),"cost"] = avgcost
    #print data["year"]
    data.loc[pd.isnull(data["year"]),"year"] = avgyear
    print data["units"]
    print 'updated data'


    min_len = 99999999999999999999
    for j in range(20):
        [temp_data,team_table,temp_routes] = iterate_apts(data,center_coords,random_function,avgyear,avgcost,est_canvas_teams,team_max)
        total_len = sum([i[-1] for i in temp_routes])/len(team_table)
        if total_len<min_len:
            min_len = total_len
            min_data = temp_data.copy()
            min_team = team_table.copy()
            min_routes = temp_routes[:]


    pdf = FPDF()

    #Scroll through each team, label apartment type, write PDFs 
    for ind,row in min_team.iterrows():
        teams = row.teams
        temp_table = min_data[min_data["team"]==ind]
        temp_table["order"] = get_order(min_routes[ind][0][1:])
        temp_table["apt_type"] = "bonus"
        temp_table["apt_sort"] = 2
        big_apts = temp_table[temp_table["units"]>=team_max]
        start = True
        for ind1,row1 in big_apts.iterrows():
            if start:
                temp_table.at[ind1,"apt_type"] = "main"
                temp_table.at[ind1,"apt_sort"] = 0
                start = False
            else:
                temp_table.at[ind1,"apt_type"] = "backup"
                temp_table.at[ind1,"apt_sort"] = 1
        temp_table = temp_table.sort_values(by=["apt_sort","order"])
        add_pages(temp_table,int(teams),pdf,ind)
    pdf.output(folder_name + "/temp_folder/pdf.pdf")

    #Email 3 pdfs to email address specified
    send_email(email,folder_name)
    print 'sent email'

    #Delete the temp folder
    shutil.rmtree(folder_name)

    return

