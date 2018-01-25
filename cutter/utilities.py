from math import radians, sin, cos, sqrt, asin
from sklearn.cluster import KMeans
import pandas as pd
import geocoder
import gmplot
import numpy as np
import csv

from fpdf import FPDF
from selenium import webdriver
import os
import time
import zipfile

import httplib2

from googleapiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from googleapiclient import errors, discovery
import mimetypes
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase



#Take a row with "LAT" and "LON" and then kwargs 'lat1' and 'lon1'
#And calculate teh haversine distance between the 2 points
def add_distance(row,**kwargs):
    #print kwargs
    lat2 = row["LAT"]
    lon2 = row["LON"]
    distance = haversine(kwargs['lat1'],kwargs['lon1'],lat2,lon2)
    return distance


#Get the distance in km between 2 lat and lon points
def haversine(lat1, lon1, lat2, lon2):
 
  R = 6372.8 # Earth radius in kilometers
 
  dLat = radians(lat2 - lat1)
  dLon = radians(lon2 - lon1)
  lat1 = radians(lat1)
  lat2 = radians(lat2)
 
  a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
  c = 2*asin(sqrt(a))
 
  return R * c


#Use the Google geocoder to get the lat and lon from an address
def get_coordinates(address):
    g = geocoder.google(address)
    return list(g.latlng)


#Take the data (master list of addresses)
#And the canvas_location (central address)
#And find the num_turfs * turf_sizes closest addresses
#Save it as new_file_name
def make_filtered_file(data,canvas_location,num_turfs,turf_sizes,new_file_name):
    #Get the coordinates of the canvas central point
    coords = get_coordinates(canvas_location,False)
    lat1 = coords[0]
    lon1 = coords[1]
    #Add a column to the list of addresses
    #That measures the distance from the center location
    data["distances"] = data.apply(add_distance,axis=1,lat1=lat1,lon1=lon1)
    #Sort the addresses by this distance
    data = data.sort_values("distances")
    #Cut the data to get the closest n addresses
    slice_data = data[:num_turfs * turf_sizes]
    #Save them as an Excel file
    slice_data.to_excel(new_file_name)



def get_coordinates(address,check_intersect=True):
    g = geocoder.google(address)
    if check_intersect:
        if not g.address:
            return []
        elif "&" in g.address:
            return list(g.latlng)
        else:
            return []
    return list(g.latlng)


#Open the list of intersections
#Load them into a dictionary
#Do it for both - so if 12th Street and Walnut Street intersect
#threshold_dict['12th Street']['Walnut Street'] will be an entry
#As will threshold_dict['Walnut Street']['12th Street']
def load_threshold_dict(lon_only=False):
    intersect_data = pd.read_csv("Intersections_1.csv")
    if lon_only:
        intersect_data = intersect_data[intersect_data["Lon"].notnull()]
    threshold_dict = {}
    for row in intersect_data.itertuples():
        s1 = row.STREET1
        s2 = row.STREET2
        if not s1 in threshold_dict:
            threshold_dict[s1] = {}
        threshold_dict[s1][s2] = None
        if not s2 in threshold_dict:
            threshold_dict[s2] = {}
        threshold_dict[s2][s1] = None
    return threshold_dict


#Take a list of data and get a list of all potential intersections
def get_potential_intersections(slice_data,km_threshold = .1,lat_threshold=.0008):
    #Load the list of already checked intersections
    threshold_dict = load_threshold_dict()
    threshold_list = {}

    #Scroll through all through potential combinations of addresses
    #Basically we're trying to estimate which streets might intersect
    #We do this by finding pairs of addresses that are close to each other
    #But on different streets
    #This is very computationally expensive
    #So let's see if we can improve it
    for i,row in enumerate(slice_data.itertuples()):
        for j,test_row in enumerate(slice_data.itertuples()):
            #Skip if the addresss are the same
            if j<=i:
                continue
            #If we've already checked the intersection don't check again
            if row.STREET in threshold_dict and test_row.STREET in threshold_dict[row.STREET]:
                continue
            #If they're the same street don't bother checking
            if row.STREET == test_row.STREET:
                continue
            #If they're not close together don't bother checking
            if abs(row.LAT - test_row.LAT) > lat_threshold:
                continue
            #Check distance between the 2 points
            #If they're not close keep going
            distance = haversine(row.LAT,row.LON,test_row.LAT,test_row.LON)
            if distance > km_threshold:
                continue
            #Sort the streets for the purpose of writing them to the dictionary
            streets = tuple(sorted([row.STREET,test_row.STREET]))
            #If this is the first time we've checked the 2 streets set their distance as the distance between the streets
            if streets not in threshold_list:
                threshold_list[streets] = distance
            #Otherwise check if these points are the 2 closest points on these 2 particular streets
            elif distance < threshold_list[streets]:
                threshold_list[streets] = distance
    return threshold_list


#Write an intersection to the master list of intersections
def write_row(row,new_file=False):
    if new_file:
        f = open('Intersections_1.csv','wb')
    else:
        f = open('Intersections_1.csv','ab')
    writer = csv.writer(f)
    writer.writerow(row)
    f.close()


#Check the thresholds for a list of addresses
def update_thresholds(slice_data):
    #Get the list of all potential intersections
    potential_intersections = get_potential_intersections(slice_data)
    num_geocodes=0
    err_count = 0
    #Scroll through the list of potential intersections
    for (i,j),k in potential_intersections.iteritems():
        num_geocodes+=1
        address = i + " and " + j +", Austin, TX"
        #Check Google's geocoder to see if it's a valid intersection
        #e.g. if we're checking 12th Street and Walnut Street
        #Check the coordinates of 12th Street & Walnut Street, Austin, TX
        #If it's not a valid intersection it won't return coordinates
        coords = get_coordinates(address)
        #Return an error message if 8 straight addresses show no intersection
        #This is a good sign that the API is maxed out
        if coords:
            err_count = 0
        else:
            err_count += 1
            if err_count >7:
                return False
        #Write the result of the check to the master list of intersections
        write_row([i,j,k] + coords)
        #Google gives us 1500 checks per day
        #If we run through them move on
        #And send a message saying to run again tomorrow
        if num_geocodes > 1500:
            return False
    return True
    

#Update the data to set the nulls to 0s
def update_slice_data_nulls(slice_data):
    slice_data.loc[slice_data["voters"].isnull(),"voters"] = 0
    slice_data.loc[slice_data["doors"].isnull(),"doors"] = 0
    return slice_data


#Add 2 columns to the list of addresses
#For the average lat and average lon of the whole street
#This will be used for the clustering algorithm
def update_slice_data_avgs(slice_data):
    avg_locs = pd.pivot_table(slice_data,values=("LAT","LON"),index=("STREET"),aggfunc=np.mean)
    avg_locs = avg_locs.rename_axis(None, axis=1).reset_index()
    avg_locs.columns = ["STREET","street_lat","street_lon"]
    avg_locs["street_lat"] = avg_locs["street_lat"]  
    avg_locs["street_lon"] = avg_locs["street_lon"] 
    slice_data_updated = slice_data.merge(avg_locs,on="STREET")
    return slice_data_updated


#Assign a cluster to each address
def update_slice_data_clusters(slice_data_updated,num_clusters):
    fit_data = slice_data_updated.loc[:,("LAT","LON","street_lat","street_lon")]
    #Use the kmeans algorithm to cluster each address into the specified # of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(fit_data)
    slice_data_updated['Cluster'] = kmeans.labels_
    return slice_data_updated


#Check that a clusters is a continuous route
#For each street check if it connects to every other street 
#Return a list of streets that don't connect
def check_cluster(df,cluster,threshold_dict):
    ret = []
    df = df[df["Cluster"]==cluster]
    df = df.reset_index()
    streets = df["STREET"].unique()
    for i in streets:
        count = checkstreet(i,streets,threshold_dict)
        if count > 0:
            ret.append([i,count])
    return ret


#For a street and a cluster
#Check if it connects to the other streets in the cluster
#using a modification of Dijkstra's algorithm
#Return the number of streets in the cluster it does not connect to
def checkstreet(street,streets,threshold_dict):
    ret = []
    unchecked = {i:999 for i in streets}
    checked = {}
    unchecked[street] = 0

    while unchecked:
        minval = 1000
        minkey = None
        for key,val in unchecked.iteritems():
            if val < minval:
                minval = val
                minkey = key

        checked[minkey] = unchecked[minkey]
        del unchecked[minkey]
        if not minkey in threshold_dict:
            continue
        to_check = set(unchecked.keys()).intersection(threshold_dict[minkey].keys())
        for i in to_check:
            distance = minval + 1
            unchecked[i] = min(distance,unchecked[i])

    for key,val in checked.iteritems():
        if val == 999:
            ret.append(street)

    return len(ret)


#Take a street that does not fit in its current cluster
#And try to find a new cluster for it
#Inputs are:
#street - name of the non-fitting street
#cur_cluster - number of the cluster it's currently in
#slice_data_updated - list of all points
#threshold_dict - list of all intersections
def new_cluster(street,cur_cluster,slice_data_updated,threshold_dict):
    #Take all addresses on the specified street in the specified cluster
    street_points = slice_data_updated.loc[(slice_data_updated["STREET"]==street) & (slice_data_updated["Cluster"]==cur_cluster),
                                ("LAT","LON")]
    #Find the mean lat and lon of that street
    street_avg = np.mean(street_points["LAT"]),np.mean(street_points["LON"])
    #Take the mean lat and lon of all the other clusters
    cluster_avgs = pd.pivot_table(slice_data_updated,index="Cluster",values=("LAT","LON"),aggfunc=np.mean)
    cluster_avgs = cluster_avgs.rename_axis(None, axis=1).reset_index()
    #Sort the rest of the clusters to find the ones closest to the specified street
    cluster_avgs["distance"] = cluster_avgs.apply(lambda row: haversine(row["LAT"],row["LON"],street_avg[0],street_avg[1]),axis=1)
    cluster_avgs = cluster_avgs.sort_values("distance")
    #Scroll through the 4 closest clusters and see if the street fits in any of them
    for count,row in enumerate(cluster_avgs.itertuples()):
        if count > 4:
            break
        #Obviously don't put the street in its same cluster
        if row.Cluster == cur_cluster:
            continue
        #df will be all the addresses in the cluster we're checking
        df = slice_data_updated[slice_data_updated["Cluster"]==row.Cluster]
        df = df.reset_index()
        #Get a list of all the streets in the cluster we're checking
        #And make sure they all connect to the street we're moving the new street to
        #If they do move the street to that cluster
        streets = df["STREET"].unique()
        if checkstreet(street,streets,threshold_dict) == 0:
            return row.Cluster
    return None


#Scroll through each cluster and remove the streets that don't connect
#Then try to move those streets to a new cluster
def update_slice_data_check_clusters(slice_data_updated,num_clusters,threshold_dict):
    #Scroll through each cluster
    for i in range(num_clusters):
        #result will be a list of streets in that cluster that don't connect to others
        result = check_cluster(slice_data_updated,i,threshold_dict)
        #If they all connect result will be blank
        #And we can continue to the next cluster
        if not result:
            continue
        #If there are streets that don't connect work through them
        while result:
            #Take the street that doesn't connect to the most other streets
            bad_street = sorted(result,key=lambda k: k[1],reverse=True)[0][0]
            #See if we can find a new cluster for that ill-fitting street
            upd_cluster = new_cluster(bad_street,i,slice_data_updated,threshold_dict)
            #If we find a new home for that street move it to that street
            #If not put it in cluster -1, which is where we keep all the streets that couldn't find a home
            if upd_cluster:
                slice_data_updated.loc[(slice_data_updated.STREET==bad_street) & (slice_data_updated.Cluster == i), "Cluster"] = upd_cluster
            else:
                slice_data_updated.loc[(slice_data_updated.STREET==bad_street) & (slice_data_updated.Cluster == i), "Cluster"] = -1
            #Check again for non-fitting streets
            #If none fit it will break the while loop and we move to the next cluster
            #If there are still streets that don't fit run this process again
            result = check_cluster(slice_data_updated,i,threshold_dict)
    return slice_data_updated
        


#Take all streets that were removed from clusters that couldn't find a new one
#And see if we can put them in a new cluster
#Checking if it connects to all the other streets
def check_bad_streets(slice_data_updated,threshold_dict):
    #Cluster -1 is all the streets that aren't a cluster
    bad_streets = slice_data_updated.loc[slice_data_updated.Cluster==-1,"STREET"].unique()
    for bad_street in bad_streets:
        #Run the new_cluster function which will return a new cluster or None
        upd_cluster = new_cluster(bad_street,-1,slice_data_updated,threshold_dict)
        #If it returns, update the street to put those addresses in a new cluster
        if upd_cluster:
            slice_data_updated.loc[(slice_data_updated.STREET==bad_street) & (slice_data_updated.Cluster == -1),"Cluster"] = upd_cluster
    return slice_data_updated



def check_split(filt_data):
    filt_data = filt_data.sort_values('doors',ascending=False).reset_index()
    max_doors = filt_data.loc[0,"doors"]
    sum_doors = sum(filt_data["doors"])
    if sum_doors - max_doors < 80:
        return False
    else:
        return True

#Check the distance between the 2 furthest apart addresses on the same street
def row_walking_distance(row):
    return haversine(row["latmin"],row["lonmin"],row["latmax"],row["lonmax"])

#Function to approximate the walking distance for a turf/cluster
#The formula is for each street takes the max and min lat and lon and gets the distance
#This will underestimate the walking distance for curvy roads
#But provides a decent estimate
def get_walking_distance(row,**kwargs):
    #Pass the list of data as a kwarg in the apply function
    slice_data_updated = kwargs['slice_data_updated']
    #df gets all the addresses in a specified cluster
    df = slice_data_updated[slice_data_updated['Cluster']==int(row['Cluster'])]
    #Get the min and max lat and lon for each street in the cluster
    f = {"LAT": ["min","max"],"LON": ["min","max"]}
    t = df.groupby(['STREET'],as_index=False).agg(f)
    t.columns = ["STREET","latmin","latmax","lonmin","lonmax"]
    #Sum the walking distance for each street in the turf
    return sum(t.apply(row_walking_distance,axis=1))


#Get row distance from a specified point
def row_distance(row,**kwargs):
        center_lat = kwargs['center_lat']
        center_lon = kwargs['center_lon']
        return haversine(row["LAT"],row["LON"],center_lat,center_lon)

#Update clusters to be ordered by distance from the center point
def update_cluster_numbers(slice_data_updated):
    #Remove the non-clusered cluster
    eligible_data = slice_data_updated[slice_data_updated["Cluster"]!=-1]
    #Take the average lat and lon of each cluster
    cluster_avgs = eligible_data.groupby("Cluster",as_index=False)["LAT","LON"].mean()
    #Make the center lat and center lon the address closest to the target
    [center_lat,center_lon] = eligible_data.loc[0,"LAT"],eligible_data.loc[0,"LON"]
    #Make a new column for each cluster - distance = it's center's distance from the overall center
    cluster_avgs["distance"] = cluster_avgs.apply(row_distance,axis=1,center_lat=center_lat,center_lon=center_lon)
    #Sort by distance from the center
    cluster_avgs = cluster_avgs.sort_values("distance")
    cluster_avgs = cluster_avgs.reset_index(drop=True)
    cluster_avgs = cluster_avgs.reset_index()
    merge_data = cluster_avgs.loc[:,("index","Cluster")]
    #Merge the new clusters with the data
    slice_data_updated = slice_data_updated.merge(merge_data,on="Cluster",how="left")
    #Delete the old cluster columns and name the added column Cluster
    slice_data_updated = slice_data_updated.drop('Cluster',1)
    slice_data_updated.columns = list(slice_data_updated.columns[:-1]) + ['Cluster']
    #Set the addresses not in a cluster to cluster -1
    slice_data_updated.loc[pd.isnull(slice_data_updated["Cluster"]),"Cluster"] = -1
    slice_data_updated["Cluster"] = slice_data_updated["Cluster"].astype(int)
    return slice_data_updated


#Take the list of addresses and return a list of turfs with stats about each turf
def get_cluster_totals(slice_data_updated):
    #Remove the addresses not assigned to a turf
    eligible_data = slice_data_updated[slice_data_updated["Cluster"]!=-1]
    #Take the average lat and lon, the number of addresses, voters and doors
    f = {"voters": ['sum',len],"doors": 'sum', "LAT": 'mean',"LON": "mean"}
    cluster_totals = eligible_data.groupby("Cluster",as_index=False).agg(f)
    cluster_totals["Cluster"] = cluster_totals["Cluster"].astype(int)
    [center_lat,center_lon] = slice_data_updated.loc[0,"LAT"],slice_data_updated.loc[0,"LON"]
    #Calculate the distance from the center
    cluster_totals["distance"] = cluster_totals.apply(row_distance,axis=1,center_lat=center_lat,center_lon=center_lon)
    #Calculate teh walking distance for each cluster
    cluster_totals["walking_distance"] = cluster_totals.apply(get_walking_distance,slice_data_updated=slice_data_updated,axis=1)
    cluster_totals.columns = ["Cluster","latmean","lonmean","doors","voters","addresses","distance","walking_distance"]
    return cluster_totals


#Split a cluster into 2
#Inputs are:
#df = total datat
#cluster = the # of the cluster we're splitting
#splits - is the number of splits (right now we're only doing 2)
def split_cluster(df,cluster,splits):
    #Make a copy so we don't endit it
    df = df.copy()
    #Filter for the addresses in the specified cluster
    filt_data = df.loc[df["Cluster"]==cluster,("LAT","LON","street_lat","street_lon","Cluster","doors")].copy()
    
    #if not check_split(filt_data):
    #    return df
    #Get the highest numbered cluster
    max_cluster = max(df["Cluster"])
    #Use kmeans to split the cluster
    kmeans = KMeans(n_clusters=splits, random_state=10).fit(filt_data)
    #Add the column label depending on where k-means put it
    filt_data["label"] = kmeans.labels_
    #Make the new clusters created by kmeans into new clusters
    for i in range(1,splits):
        filt_data.loc[filt_data["label"]==i,"Cluster"] = max_cluster + i
    df.update(filt_data["Cluster"])
    return df



def top_street(slice_data_updated,cluster):
    filt_data = slice_data_updated.loc[slice_data_updated["Cluster"]==cluster,:]
    agg_data = filt_data.groupby("STREET").agg(len).reset_index()
    street = agg_data.loc[0,"STREET"]
    return street


#Take a cluster and see if i tcan be merged with another cluster
#inputs are:
#cluster_totals = list of all clusters
#slice_data_updated = list of all addresses
#cluster = #of the cluster we're trying to merge
#threshold_dict = list of all intersections
#turf_size - desired turf size
#missing clusters - clusters that already got merged - don't try to merge with them
def new_whole_cluster(cluster_totals,slice_data_updated,cluster,threshold_dict,turf_size,missing_clusters):
    #Get the # of doors in the specified cluster
    cluster_doors = cluster_totals.loc[cluster_totals["Cluster"]==cluster,"doors"]
    cluster_doors = cluster_doors.iloc[0]
    #Get the walking distance of the specified cluster
    cluster_walking_distance = cluster_totals.loc[cluster_totals["Cluster"]==cluster,"walking_distance"]
    cluster_walking_distance = cluster_walking_distance.iloc[0]
    #Get the mean lat and lon of the specified cluster
    cluster_lat = cluster_totals.loc[cluster_totals["Cluster"]==cluster,"latmean"].iloc[0]
    cluster_lon = cluster_totals.loc[cluster_totals["Cluster"]==cluster,"lonmean"].iloc[0]
    street_avg = cluster_lat,cluster_lon
    #Get the average lat and lon of each cluster
    #And sort by distance to the cluster we're trying to merge
    cluster_avgs = pd.pivot_table(slice_data_updated,index="Cluster",values=("LAT","LON"),aggfunc=np.mean)
    cluster_avgs = cluster_avgs.rename_axis(None, axis=1).reset_index()
    cluster_avgs["distance"] = cluster_avgs.apply(lambda row: haversine(row["LAT"],row["LON"],street_avg[0],street_avg[1]),axis=1)
    cluster_avgs = cluster_avgs.sort_values("distance")
    #Get the most common street in the cluster we're trying to merge
    street = top_street(slice_data_updated,cluster)
    #Scroll through all other clusters
    for count,row in enumerate(cluster_avgs.itertuples()):
        #Only try the 4 clusters closest to the specified cluster
        if count > 4:
            break
        #Skip the missing clusters list
        if row.Cluster in missing_clusters:
            continue
        #Cluster -1 isn't an actual cluster
        if row.Cluster == -1:
            continue
        #Don't merge a cluster with itself
        if row.Cluster == cluster:
            continue
        #Filter the data for the cluster we're scrolling through    
        df = slice_data_updated[slice_data_updated["Cluster"]==row.Cluster]
        df_doors = sum(df["doors"])
        #Get walking distance of the specified cluster
        walking_distance = cluster_totals.loc[cluster_totals["Cluster"]==row.Cluster,"walking_distance"]
        walking_distance = walking_distance.iloc[0]

        #Use the formula to find the max numbers of doors in a cluster
        max_doors = 2.4 * turf_size - 30 * walking_distance - 30 * cluster_walking_distance
        #If merging these 2 clusters if over the max # of doors we can't merge
        if df_doors + cluster_doors >= max_doors:
            continue
        df = df.reset_index()
        #Get unique streets in 2nd cluster
        streets = df["STREET"].unique()
        #If main street connects to all streets in the proposed cluster they can merge
        if checkstreet(street,streets,threshold_dict) == 0:
            return row.Cluster
    return None


#Take a list of addresses and make an html file with the data points plotted
def make_html_file(df,folder):
    #Find the center of addresses specified
    center_lat,center_lon = np.mean(df["LAT"]),np.mean(df["LON"])
    #Make a Google map centered on the specified address
    gmap = gmplot.GoogleMapPlotter(center_lat,center_lon,16)
    #Split the addresses into 4 categories
    df_no_voters = df[df["voters"]==0]
    df_some_voters = df[(df["voters"]>0) & (df["voters"]<4)]
    df_many_voters = df[(df["voters"]>3) & (df["voters"]<10)]
    df_ton_voters = df[df["voters"]>9]
    #Plot the 4 categories on the map
    gmap.scatter(df_no_voters["LAT"], df_no_voters["LON"], color="green",marker=False,size=10)
    gmap.scatter(df_some_voters["LAT"], df_some_voters["LON"], color="black",marker=False,size=8)
    gmap.scatter(df_many_voters["LAT"], df_many_voters["LON"], color="black",marker=False,size=15)
    gmap.scatter(df_ton_voters["LAT"], df_ton_voters["LON"], color="black",marker=False,size=23)
    #Save the map as an HTML file
    gmap.draw(folder + "/temp_map.html")


#Add the text to a PDF page we're making
def text_page(pdf,cluster,street_list,doors,voters):
    #Add a page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 50)
    #Write the team name (based on cluster #)
    pdf.cell(30, 20, 'Team # ' + str(cluster),ln=2)
    pdf.set_font('Arial', 'B', 24)
    #Specify the # of doors and voters
    pdf.cell(20, 10, str(doors) + " registered doors",ln=2)
    pdf.cell(20, 10, str(voters) + " registered voters",ln=2)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    #Write each street in the turf
    for i in street_list:
        pdf.cell(16, 8, i,ln=2)
    #Make dots to specify what each means    
    pdf.set_fill_color(r = 0)
    pdf.ellipse(pdf.get_x() + 3,pdf.get_y() + 3,3,3,style='F')
    pdf.cell(8,3,"")
    pdf.cell(20, 10, "Address w/ voter",ln=1)
    pdf.ellipse(pdf.get_x() + 3,pdf.get_y() + 3,6,6,style='F')
    pdf.cell(12,3,"")
    pdf.cell(20, 10, "Address w/ many voters",ln=1)
    pdf.set_fill_color(r = 0, g = 255, b = 0)
    pdf.ellipse(pdf.get_x() + 3,pdf.get_y() + 3,3,3,style='F')
    pdf.cell(8,3,"")
    pdf.cell(20, 10, "Address w/ no voters",ln=1)


#Get a list of streets and ranges of address in a specified turf
def get_street_list(df):
    streets = pd.pivot_table(df,index="STREET",values="NUMBER",aggfunc=(min,max))
    streets = streets.rename_axis(None, axis=1).reset_index()
    ret = []
    for row in streets.itertuples():
        ret.append(str(row.min) + " " + row.STREET + " to " + str(row.max) + " " + row.STREET)
    return ret


#Convert the html file into a png file
#This is the most time-consuming part of the process
#We are opening it in a browser and taking a screenshot
#So if there's a better way that is preferable
def make_img_file(cluster,folder):
    #Have a virtual Chrome driver open the html file
    driver = webdriver.Chrome()  
    driver.get("file:///home/mike/canvas_cutting/canvas_cutting/{folder}/temp_map.html".format(folder=folder))
    #Wait 4 seconds for the page to load
    time.sleep(4)      
    #Save the screenshot as a file
    driver.get_screenshot_as_file("{folder}/temp_map_{cluster}.png".format(cluster=cluster,folder=folder))
    driver.quit()


#Add an image to a specified pdf file
def add_img(pdf,cluster,folder,w=None):
    if w:
        pdf.image("{folder}/temp_map_{cluster}.png".format(cluster=cluster,folder=folder),w=w)
    else:
        pdf.image("{folder}/temp_map_{cluster}.png".format(cluster=cluster,folder=folder))


#Figure out how many rows to make 
#To be safe we double the number of registered doors at an address
#i.e. if an apartment has 14 registered doors we leave 28 spaces
#as not every resident will be registered
def num_rows(row):
    return min(2,2*int(row.doors)/2)


#For each address make cells that will be on the list of addresses given to the canvassers
#Each address gets at least a white row and a gray row
def write_address_rows(pdf,address):
    pdf.set_font('Times','',13.0) 
    th = pdf.font_size
    pdf.cell(4, th, address, border=1)
    pdf.cell(1.25, th, "", border=1)
    pdf.cell(1, th, "", border=1)
    pdf.cell(.75, th, "", border=1)
    pdf.cell(3, th, "", border=1)
    pdf.ln(th)
    pdf.cell(4, th, "", border=1,fill=True)
    pdf.cell(1.25, th, "", border=1,fill=True)
    pdf.cell(1, th, "", border=1,fill=True)
    pdf.cell(.75, th, "", border=1,fill=True)
    pdf.cell(3, th, "", border=1,fill=True)
    pdf.ln(th)

#Write the first row on a new page - it will have the headers
def write_header_row(pdf):
    pdf.set_font('Times','',13.0) 
    th = pdf.font_size
    pdf.cell(4, th, "Address", border=1)
    pdf.cell(1.25, th, "Unit #", border=1)
    pdf.cell(1, th, "Residence?", border=1)
    pdf.cell(.75, th, "Home?", border=1)
    pdf.cell(3, th, "Notes", border=1)
    pdf.ln(th)

#On a new page list the team #
#And add a header row
def new_page(pdf,cluster,cont=False):
    pdf.add_page()
    
    header_text = 'Turf # ' + str(cluster)
    if cont:
        header_text += " (Continued)"
    
    pdf.set_font('Times','B',14.0) 
    pdf.cell(10, 0.0, header_text, align='C')
    pdf.set_font('Times','',13.0) 
    pdf.ln(0.5)
    
    write_header_row(pdf)
    
    return pdf


#For each address write rows onto a sheet
#pdf - the pdf file
#row - a row of data represeting an address
#cluster - the number of the cluster
#row_count - how far down the sheet we are
def write_address(pdf,row,cluster,row_count):
    address = row.address
    #Even an address with no registered doors will get printed
    print_doors = max(1,row.doors)
    #Start at the current row on the page and add the number of doors
    for i in range(row_count,row_count + print_doors):
        #Print 2 rows for each door
        #And when we're done a page make a new page
        if row_count > 0 and row_count % 20 == 0:
            pdf = new_page(pdf,cluster,cont=True)
        write_address_rows(pdf,address)
        row_count += 1
    #Return how far down the page we are
    return row_count


#For a cluster make a new page and then write each address to that page
def write_cluster(pdf,data,cluster):
    new_page(pdf,cluster)
    filt_data = data[data["Cluster"]==cluster]
    #Sort addresses by street and number
    #This will make it easy to find the needed address
    filt_data = filt_data.sort_values(by=["STREET","NUMBER"])
    row_count = 0
    for row in filt_data.itertuples():
        row_count = write_address(pdf,row,cluster,row_count)
    #Add a blank page at the end
    #We found that when double sided printing otherwise we'd end up with cluster 2 and cluster 3 on a single page
    pdf.add_page()

#Write the sheet that we will use to assign turfs
def write_assign_sheet(pdf,row,turf_size):
    pdf.set_font('Times','B',10.0) 
    th = pdf.font_size
    pdf.cell(1.75, th, "Team # " + str(row.Cluster), border=1)
    pdf.set_font('Times','',10.0) 
    pdf.cell(1.75, th, str(row.doors) + " registered doors", border=1)
    pdf.cell(1.75, th, str(row.voters) + " registered voters", border=1)
    pdf.cell(1.75, th, str(row.walking_distance) + " km", border=1)
    pdf.ln(th)
    pdf.cell(1.75, th, "Team Captain", border=1)
    pdf.cell(1.75, th, "Team Member 2", border=1)
    print row
    print turf_size
    print row.doors > 1.6 * turf_size - 30 * row.walking_distance
    #Depending on turf size and walking distance assign a team 2 members or 4 members
    if row.doors > 1.6 * turf_size - 30 * row.walking_distance:
        pdf.cell(1.75, th, "Team Member 3", border=1)
        pdf.cell(1.75, th, "Team Member 4", border=1)
    pdf.ln(th)
    pdf.cell(1.75, th, "", border=1)
    pdf.cell(1.75, th, "", border=1)
    if row.doors > 1.6 * turf_size - 30 * row.walking_distance:
        pdf.cell(1.75, th, "", border=1)
        pdf.cell(1.75, th, "", border=1)
    pdf.ln(th)


#Get credentials for sending an email
def get_credentials():
    flags = None
    SCOPES = 'https://www.googleapis.com/auth/gmail.send'
    CLIENT_SECRET_FILE = 'client_secret.json'
    APPLICATION_NAME = 'Turf Clustering App'
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'gmail-python-email-send.json')
    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        flags = tools.argparser.parse_args(args=[])
        credentials = tools.run_flow(flow, store,flags)
        print('Storing credentials to ' + credential_path)
    return credentials


#Send an email
def SendMessage(sender, to, subject, msgHtml, msgPlain, attachmentFile=None):
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('gmail', 'v1', http=http)
    if attachmentFile:
        message1 = createMessageWithAttachment(sender, to, subject, msgHtml, msgPlain, attachmentFile)
    else: 
        message1 = CreateMessageHtml(sender, to, subject, msgHtml, msgPlain)
    result = SendMessageInternal(service, "me", message1)
    return result

#Function for sending an email
def SendMessageInternal(service, user_id, message):
    try:
        message = (service.users().messages().send(userId=user_id, body=message).execute())
        print('Message Id: %s' % message['id'])
        return message
    except errors.HttpError as error:
        print('An error occurred: %s' % error)
        return "Error"
    return "OK"


#Function for creating email html
def CreateMessageHtml(sender, to, subject, msgHtml, msgPlain):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = to
    msg.attach(MIMEText(msgPlain, 'plain'))
    msg.attach(MIMEText(msgHtml, 'html'))
    return {'raw': base64.urlsafe_b64encode(msg.as_string())}

#Create an email message with an attachment
def createMessageWithAttachment(
    sender, to, subject, msgHtml, msgPlain, attachmentFile):
    """Create a message for an email.

    Args:
      sender: Email address of the sender.
      to: Email address of the receiver.
      subject: The subject of the email message.
      msgHtml: Html message to be sent
      msgPlain: Alternative plain text message for older email clients          
      attachmentFile: The path to the file to be attached.

    Returns:
      An object containing a base64url encoded email object.
    """
    message = MIMEMultipart('mixed')
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject

    messageA = MIMEMultipart('alternative')
    messageR = MIMEMultipart('related')

    messageR.attach(MIMEText(msgHtml, 'html'))
    messageA.attach(MIMEText(msgPlain, 'plain'))
    messageA.attach(messageR)

    message.attach(messageA)

    print("create_message_with_attachment: file: %s" % attachmentFile)
    content_type, encoding = mimetypes.guess_type(attachmentFile)

    if content_type is None or encoding is not None:
        content_type = 'application/octet-stream'
    main_type, sub_type = content_type.split('/', 1)
    if main_type == 'text':
        fp = open(attachmentFile, 'rb')
        msg = MIMEText(fp.read(), _subtype=sub_type)
        fp.close()
    elif main_type == 'image':
        fp = open(attachmentFile, 'rb')
        msg = MIMEImage(fp.read(), _subtype=sub_type)
        fp.close()
    elif main_type == 'audio':
        fp = open(attachmentFile, 'rb')
        msg = MIMEAudio(fp.read(), _subtype=sub_type)
        fp.close()
    else:
        fp = open(attachmentFile, 'rb')
        msg = MIMEBase(main_type, sub_type)
        msg.set_payload(fp.read())
        fp.close()
    filename = os.path.basename(attachmentFile)
    msg.add_header('Content-Disposition', 'attachment', filename=filename)
    message.attach(msg)

    return {'raw': base64.urlsafe_b64encode(message.as_string())}

#Convert a folder to a zip file
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

#Create the zip file to send out
def zip_folder(folder):
    zipf = zipfile.ZipFile(folder + '/temp_email.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir(folder + '/temp_folder', zipf)
    zipf.close()

#Send the email
def send_email(to,folder_path):
    zip_folder(folder_path)
    SendMessage("turfclusteringapp@gmail.com", to, "Here are your turfs", "", "Here is some stuff", attachmentFile=folder_path + '/temp_email.zip')

#Send an error email if a process fails
def send_error_email(to):
    SendMessage("turfclusteringapp@gmail.com", to, "Error making your turfs", "", "We were not able to generate your turfs. Unfortunately we hit the API limit checking addresses to make your turfs. Please run again in 24 hours.")
